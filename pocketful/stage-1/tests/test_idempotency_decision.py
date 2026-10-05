"""Tests for the idempotency decision logic (PDF-32)."""

import json
import sqlite3
import threading

import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.idempotency.decision import (
    ClaimResult,
    Decision,
    claim,
    decide,
    update_response,
)
from app.idempotency.repository import Outcome
from app.sqlite_utils.transaction import write_transaction


@pytest.fixture
def isolated_conn(tmp_path):
    """Provide an isolated SQLite connection for the test.

    Uses tmp_path so each test gets its own DB file and never touches the
    session-level db_path used by the FastAPI integration tests in conftest.
    """
    path = str(tmp_path / "decision.db")
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")

    conn.execute(
        "CREATE TABLE users ("
        "  id TEXT PRIMARY KEY,"
        "  email TEXT NOT NULL UNIQUE,"
        "  password_hash TEXT NOT NULL,"
        "  display_name TEXT NOT NULL,"
        "  handle TEXT NOT NULL UNIQUE,"
        "  balance INTEGER NOT NULL DEFAULT 0,"
        "  currency TEXT NOT NULL,"
        "  minor_units INTEGER NOT NULL,"
        "  created_at TEXT NOT NULL"
        ")"
    )

    with write_transaction(conn) as cur:
        cur.execute(
            "INSERT INTO users(id, email, password_hash, display_name, handle, balance, currency, minor_units, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("u_alice", "a@x", "x", "Alice", "alice", 0, "EUR", 2, "2026-01-01T00:00:00+00:00"),
        )
        cur.execute(
            "INSERT INTO users(id, email, password_hash, display_name, handle, balance, currency, minor_units, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("u_bob", "b@x", "x", "Bob", "bob", 0, "EUR", 2, "2026-01-01T00:00:00+00:00"),
        )

    from app.idempotency.schema import apply_schema
    apply_schema(conn)

    yield conn
    conn.close()


def _payment_body(amount: int = 1500, note: str = "dinner", visibility: str = "public") -> dict:
    return {"to_handle": "bob", "amount": amount, "note": note, "visibility": visibility}


def _seed_response_body(payload: dict | None = None) -> str:
    if payload is None:
        payload = {"payment_id": "p_42", "amount": 1500}
    return json.dumps(payload)


def _seed(conn, user_id: str, key: str, body: dict, response_body: str | None = None) -> None:
    """Helper that pre-populates the store with a known record."""
    from app.idempotency.repository import store

    if response_body is None:
        response_body = _seed_response_body()
    store(
        conn,
        user_id=user_id,
        endpoint="POST /payments",
        key=key,
        body=body,
        response_status=201,
        response_body=response_body,
        created_at="2026-09-24T11:04:03+00:00",
    )


class TestDecideMissing:
    def test_missing_when_key_unseen(self, isolated_conn):
        decision = decide(
            isolated_conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="never-seen",
            body=_payment_body(),
        )
        assert isinstance(decision, Decision)
        assert decision.is_missing
        assert decision.outcome == Outcome.MISSING
        assert decision.record is None
        assert decision.response_status is None
        assert decision.response_body is None

    def test_missing_does_not_write(self, isolated_conn):
        """A MISSING decision must not produce any row in the store."""
        decide(
            isolated_conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="no-write",
            body=_payment_body(),
        )
        n = isolated_conn.execute(
            "SELECT COUNT(*) AS n FROM idempotency_keys WHERE key = ?",
            ("no-write",),
        ).fetchone()["n"]
        assert n == 0


class TestDecideReplay:
    def test_replay_returns_stored_response(self, isolated_conn):
        body = _payment_body()
        response_body = _seed_response_body({"payment_id": "p_99", "amount": 1500})
        _seed(isolated_conn, user_id="u_alice", key="replay-key", body=body, response_body=response_body)

        decision = decide(
            isolated_conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="replay-key",
            body=body,
        )

        assert decision.is_replay
        assert decision.outcome == Outcome.REPLAY
        assert decision.record is not None
        assert decision.response_status == 201
        assert decision.response_body == response_body
        assert decision.response_payload == {"payment_id": "p_99", "amount": 1500}

    def test_replay_with_different_key_order_same_logical_body(self, isolated_conn):
        """Different JSON key order must still produce REPLAY (canonical hash)."""
        body = _payment_body()
        _seed(isolated_conn, user_id="u_alice", key="key-order", body=body)

        shuffled = {
            "visibility": "public",
            "note": "dinner",
            "amount": 1500,
            "to_handle": "bob",
        }

        decision = decide(
            isolated_conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="key-order",
            body=shuffled,
        )
        assert decision.is_replay

    def test_replay_key_stored_verbatim(self, isolated_conn):
        """The key in the stored record is byte-for-byte what was passed in."""
        body = _payment_body()
        raw_key = "client-chosen-2f9c1a!#%&*+="
        _seed(isolated_conn, user_id="u_alice", key=raw_key, body=body)

        decision = decide(
            isolated_conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key=raw_key,
            body=body,
        )
        assert decision.is_replay
        assert decision.record is not None
        assert decision.record.key == raw_key
        # The key is NOT replaced by its hash anywhere in the record
        assert decision.record.key != decision.record.request_body_hash


class TestDecideConflict:
    def test_conflict_when_same_key_different_body(self, isolated_conn):
        body_a = _payment_body(amount=1500)
        _seed(isolated_conn, user_id="u_alice", key="conflict-key", body=body_a)

        body_b = _payment_body(amount=2500)  # different amount => different hash

        decision = decide(
            isolated_conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="conflict-key",
            body=body_b,
        )

        assert decision.is_conflict
        assert decision.outcome == Outcome.CONFLICT
        assert decision.record is not None
        # The stored hash reflects the *original* body, not the new one
        assert decision.record.request_body_hash != _body_hash(body_b)
        # The new body must not be mutated by the decision call
        assert body_b == _payment_body(amount=2500)


class TestScoping:
    def test_same_key_different_user_is_missing(self, isolated_conn):
        body = _payment_body()
        _seed(isolated_conn, user_id="u_alice", key="shared-key", body=body)

        decision = decide(
            isolated_conn,
            user_id="u_bob",  # different user
            endpoint="POST /payments",
            key="shared-key",
            body=body,
        )
        assert decision.is_missing
        assert decision.record is None

    def test_same_key_same_user_different_endpoint_is_missing(self, isolated_conn):
        body = _payment_body()
        _seed(isolated_conn, user_id="u_alice", key="multi-endpoint-key", body=body)

        decision = decide(
            isolated_conn,
            user_id="u_alice",
            endpoint="POST /requests",  # different endpoint
            key="multi-endpoint-key",
            body=body,
        )
        assert decision.is_missing
        assert decision.record is None

    def test_conflict_is_also_user_scoped(self, isolated_conn):
        """A different body sent by a different user with the same key is MISSING,
        not CONFLICT — the conflict is per-user."""
        body_a = _payment_body(amount=1500)
        body_b = _payment_body(amount=2500)
        _seed(isolated_conn, user_id="u_alice", key="cross-user", body=body_a)

        decision = decide(
            isolated_conn,
            user_id="u_bob",  # different user
            endpoint="POST /payments",
            key="cross-user",
            body=body_b,
        )
        assert decision.is_missing


class TestClaim:
    def test_claim_winner_records_placeholder(self, isolated_conn):
        body = _payment_body()
        result = claim(
            isolated_conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="claim-key",
            body=body,
            response_status=201,
            response_body="",
            created_at="2026-09-24T11:04:03+00:00",
        )

        assert isinstance(result, ClaimResult)
        assert result.is_winner
        assert result.outcome == Outcome.MISSING
        assert result.record is not None
        assert result.record.user_id == "u_alice"
        assert result.record.key == "claim-key"
        assert result.record.response_status == 201
        assert result.record.response_body == ""
        assert result.record.request_body_hash == _body_hash(body)

    def test_claim_loser_returns_replay_when_body_matches(self, isolated_conn):
        body = _payment_body()
        _seed(isolated_conn, user_id="u_alice", key="replay-claim", body=body)

        result = claim(
            isolated_conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="replay-claim",
            body=body,
            response_status=201,
            response_body="",
            created_at="2026-09-24T11:04:03+00:00",
        )

        assert not result.is_winner
        assert result.is_replay
        assert result.outcome == Outcome.REPLAY
        assert result.record is not None
        assert result.response_status == 201
        assert result.response_payload == {"payment_id": "p_42", "amount": 1500}

    def test_claim_loser_returns_conflict_when_body_differs(self, isolated_conn):
        body_a = _payment_body(amount=1500)
        _seed(isolated_conn, user_id="u_alice", key="conflict-claim", body=body_a)

        body_b = _payment_body(amount=2500)

        result = claim(
            isolated_conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="conflict-claim",
            body=body_b,
            response_status=201,
            response_body="",
            created_at="2026-09-24T11:04:03+00:00",
        )

        assert not result.is_winner
        assert result.is_conflict
        assert result.outcome == Outcome.CONFLICT
        assert result.record is not None

    def test_claim_uses_canonical_body_hash(self, isolated_conn):
        """The hash stored by claim() must come from canonical_body_hash."""
        body = {"b": 2, "a": 1}
        result = claim(
            isolated_conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="hash-key",
            body=body,
            response_status=201,
            response_body="",
            created_at="2026-09-24T11:04:03+00:00",
        )
        assert result.is_winner
        assert result.record is not None
        # canonical_body_hash sorts keys, so the hash of {"b":2,"a":1} equals
        # the hash of {"a":1,"b":2}
        from app.idempotency.canonical_hash import canonical_body_hash
        assert result.record.request_body_hash == canonical_body_hash({"a": 1, "b": 2})

    def test_claim_key_preserved_verbatim(self, isolated_conn):
        raw_key = "raw-key-!@#$%^&*()"
        body = _payment_body()
        result = claim(
            isolated_conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key=raw_key,
            body=body,
            response_status=201,
            response_body="",
            created_at="2026-09-24T11:04:03+00:00",
        )
        assert result.is_winner
        assert result.record is not None
        assert result.record.key == raw_key


class TestUpdateResponse:
    def test_update_response_enriches_record(self, isolated_conn):
        body = _payment_body()
        result = claim(
            isolated_conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="update-key",
            body=body,
            response_status=201,
            response_body="",
            created_at="2026-09-24T11:04:03+00:00",
        )
        assert result.is_winner

        final = json.dumps({"payment_id": "p_77", "amount": 1500, "note": "dinner"})
        update_response(
            isolated_conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="update-key",
            response_status=201,
            response_body=final,
        )

        decision = decide(
            isolated_conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="update-key",
            body=body,
        )
        assert decision.is_replay
        assert decision.response_body == final
        assert decision.response_payload == {
            "payment_id": "p_77",
            "amount": 1500,
            "note": "dinner",
        }


class TestEndToEnd:
    def test_first_use_then_replay_then_conflict(self, isolated_conn):
        """End-to-end: first MISSING + claim -> replay -> conflict."""
        conn = isolated_conn
        body = _payment_body()

        # 1) First use: MISSING
        d1 = decide(conn, user_id="u_alice", endpoint="POST /payments", key="e2e", body=body)
        assert d1.is_missing

        # 2) Claim, perform operation, update response
        claim(
            conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="e2e",
            body=body,
            response_status=201,
            response_body="",
            created_at="2026-09-24T11:04:03+00:00",
        )
        update_response(
            conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="e2e",
            response_status=201,
            response_body=_seed_response_body({"payment_id": "p_77"}),
        )

        # 3) Replay: same body
        d2 = decide(conn, user_id="u_alice", endpoint="POST /payments", key="e2e", body=body)
        assert d2.is_replay
        assert d2.response_payload == {"payment_id": "p_77"}

        # 4) Conflict: same key, different body
        d3 = decide(conn, user_id="u_alice", endpoint="POST /payments", key="e2e", body=_payment_body(amount=9999))
        assert d3.is_conflict


def _body_hash(body: dict) -> str:
    from app.idempotency.canonical_hash import canonical_body_hash
    return canonical_body_hash(body)