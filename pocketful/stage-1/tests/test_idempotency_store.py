"""Tests for the idempotency store schema and repository."""

import json
import sqlite3
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.idempotency.repository import (
    IdempotencyRecord,
    Outcome,
    lookup,
    store,
    update_response,
)
from app.sqlite_utils.transaction import write_transaction


@pytest.fixture
def isolated_conn(tmp_path):
    """Provide an isolated SQLite connection for the test.

    Uses tmp_path so each test gets its own DB file and never touches the
    session-level db_path used by the FastAPI integration tests in conftest.
    """
    path = str(tmp_path / "idem.db")
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")

    # Minimal schema to make the FK on user_id resolve. The table under
    # test is created by the repository's schema module via apply_schema().
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


def _post_body(amount: int = 1500, note: str = "dinner", visibility: str = "public") -> dict:
    return {"to_handle": "bob", "amount": amount, "note": note, "visibility": visibility}


class TestStore:
    def test_store_persists_key_verbatim(self, isolated_conn):
        """The idempotency key must be stored exactly as provided, not as its hash."""
        conn = isolated_conn
        store(
            conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="client-key-2f9c1a",
            body=_post_body(),
            response_status=201,
            response_body="",
            created_at="2026-09-24T11:04:03+00:00",
        )

        row = conn.execute(
            "SELECT key, request_body_hash FROM idempotency_keys "
            "WHERE user_id = ? AND endpoint = ? AND key = ?",
            ("u_alice", "POST /payments", "client-key-2f9c1a"),
        ).fetchone()

        assert row is not None
        assert row["key"] == "client-key-2f9c1a"
        # Sanity: request_body_hash is a separate column, not the key itself
        assert row["request_body_hash"] != row["key"]
        assert len(row["request_body_hash"]) == 64  # SHA-256 hex digest

    def test_store_uses_canonical_body_hash(self, isolated_conn):
        """The stored request_body_hash is the canonical hash of the body."""
        from app.idempotency.canonical_hash import canonical_body_hash

        conn = isolated_conn
        body = _post_body()
        expected_hash = canonical_body_hash(body)

        store(
            conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="key-1",
            body=body,
            response_status=201,
            response_body="",
            created_at="2026-09-24T11:04:03+00:00",
        )

        row = conn.execute(
            "SELECT request_body_hash FROM idempotency_keys "
            "WHERE user_id = ? AND endpoint = ? AND key = ?",
            ("u_alice", "POST /payments", "key-1"),
        ).fetchone()
        assert row["request_body_hash"] == expected_hash

    def test_store_same_key_different_user_isolated(self, isolated_conn):
        """Two distinct users may use the same idempotency key string."""
        conn = isolated_conn
        body_a = _post_body(amount=1500)
        body_b = _post_body(amount=2500)

        store(
            conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="shared-key",
            body=body_a,
            response_status=201,
            response_body="",
            created_at="2026-09-24T11:04:03+00:00",
        )
        store(
            conn,
            user_id="u_bob",
            endpoint="POST /payments",
            key="shared-key",
            body=body_b,
            response_status=201,
            response_body="",
            created_at="2026-09-24T11:05:00+00:00",
        )

        rows = conn.execute(
            "SELECT user_id, request_body_hash FROM idempotency_keys WHERE key = ?",
            ("shared-key",),
        ).fetchall()
        assert len(rows) == 2
        user_ids = {row["user_id"] for row in rows}
        assert user_ids == {"u_alice", "u_bob"}

    def test_store_same_key_same_user_different_endpoint(self, isolated_conn):
        """The same key on a different endpoint is a distinct record."""
        conn = isolated_conn
        body = _post_body()

        store(
            conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="multi-endpoint-key",
            body=body,
            response_status=201,
            response_body="",
            created_at="2026-09-24T11:04:03+00:00",
        )
        store(
            conn,
            user_id="u_alice",
            endpoint="POST /requests",
            key="multi-endpoint-key",
            body=body,
            response_status=201,
            response_body="",
            created_at="2026-09-24T11:05:00+00:00",
        )

        rows = conn.execute(
            "SELECT endpoint FROM idempotency_keys WHERE key = ? ORDER BY endpoint",
            ("multi-endpoint-key",),
        ).fetchall()
        assert [row["endpoint"] for row in rows] == [
            "POST /payments",
            "POST /requests",
        ]


class TestLookup:
    def test_lookup_missing_returns_missing(self, isolated_conn):
        conn = isolated_conn
        outcome, record = lookup(
            conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="never-seen",
            body=_post_body(),
        )
        assert outcome == Outcome.MISSING
        assert record is None

    def test_lookup_replay_returns_record(self, isolated_conn):
        """Same key + same canonical body returns REPLAY with stored record."""
        conn = isolated_conn
        body = _post_body()
        response_body = json.dumps({"payment_id": "p_1"})

        store(
            conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="replay-key",
            body=body,
            response_status=201,
            response_body=response_body,
            created_at="2026-09-24T11:04:03+00:00",
        )

        # Equivalent body with different key order must still produce REPLAY
        shuffled_body = {"visibility": "public", "note": "dinner", "amount": 1500, "to_handle": "bob"}

        outcome, record = lookup(
            conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="replay-key",
            body=shuffled_body,
        )

        assert outcome == Outcome.REPLAY
        assert record is not None
        assert record.user_id == "u_alice"
        assert record.key == "replay-key"
        assert record.response_status == 201
        assert record.response_body == response_body

    def test_lookup_conflict_returns_conflict(self, isolated_conn):
        """Same key + different canonical body returns CONFLICT (no body mutation)."""
        conn = isolated_conn
        body_a = _post_body(amount=1500)
        store(
            conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="conflict-key",
            body=body_a,
            response_status=201,
            response_body="",
            created_at="2026-09-24T11:04:03+00:00",
        )

        body_b = _post_body(amount=2500)  # different amount -> different hash

        outcome, record = lookup(
            conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="conflict-key",
            body=body_b,
        )

        assert outcome == Outcome.CONFLICT
        assert record is not None
        assert record.request_body_hash != _diff_hash(body_b)
        # Body of the conflicting lookup must NOT have been mutated
        assert body_b == _post_body(amount=2500)

    def test_lookup_scoped_by_user(self, isolated_conn):
        """A key collision between users is independent: each user has its own record."""
        conn = isolated_conn
        body = _post_body()

        store(
            conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="cross-user-key",
            body=body,
            response_status=201,
            response_body="alice-response",
            created_at="2026-09-24T11:04:03+00:00",
        )

        # Bob looks up the same key — must MISS, not REPLAY or CONFLICT
        outcome_bob, record_bob = lookup(
            conn,
            user_id="u_bob",
            endpoint="POST /payments",
            key="cross-user-key",
            body=body,
        )
        assert outcome_bob == Outcome.MISSING
        assert record_bob is None

        # Alice still sees her record
        outcome_alice, record_alice = lookup(
            conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="cross-user-key",
            body=body,
        )
        assert outcome_alice == Outcome.REPLAY
        assert record_alice is not None
        assert record_alice.user_id == "u_alice"


class TestUpdateResponse:
    def test_update_response_modifies_existing_record(self, isolated_conn):
        conn = isolated_conn
        body = _post_body()
        store(
            conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="update-key",
            body=body,
            response_status=201,
            response_body="",
            created_at="2026-09-24T11:04:03+00:00",
        )

        update_response(
            conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="update-key",
            response_status=201,
            response_body='{"payment_id": "p_42"}',
        )

        outcome, record = lookup(
            conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="update-key",
            body=body,
        )
        assert outcome == Outcome.REPLAY
        assert record is not None
        assert record.response_body == '{"payment_id": "p_42"}'


class TestSchema:
    def test_primary_key_is_composite(self, isolated_conn):
        """The PRIMARY KEY is (user_id, endpoint, key) per spec §7."""
        conn = isolated_conn
        row = conn.execute(
            "SELECT name FROM pragma_table_info('idempotency_keys') "
            "WHERE pk > 0 ORDER BY pk"
        ).fetchall()
        pk_cols = [r["name"] for r in row]
        assert pk_cols == ["user_id", "endpoint", "key"]

    def test_unique_constraint_enforced(self, isolated_conn):
        """Inserting two records for the same (user, endpoint, key) must fail."""
        conn = isolated_conn
        body = _post_body()
        store(
            conn,
            user_id="u_alice",
            endpoint="POST /payments",
            key="dup-key",
            body=body,
            response_status=201,
            response_body="",
            created_at="2026-09-24T11:04:03+00:00",
        )
        with pytest.raises(sqlite3.IntegrityError):
            store(
                conn,
                user_id="u_alice",
                endpoint="POST /payments",
                key="dup-key",
                body=body,
                response_status=201,
                response_body="",
                created_at="2026-09-24T11:05:00+00:00",
            )


def _diff_hash(body: dict) -> str:
    from app.idempotency.canonical_hash import canonical_body_hash
    return canonical_body_hash(body)


class TestConcurrent:
    def test_concurrent_store_only_one_row_persists(self, isolated_conn):
        """Concurrent store() calls for the same key: at most one row persists.

        SQLite serializes writes through BEGIN IMMEDIATE; the first thread
        to insert wins, the rest either fail with IntegrityError (after a
        later re-run of lookup) or with OperationalError on busy_timeout.
        Either way, exactly one row should remain in the table for the
        contested (user, endpoint, key) tuple.
        """
        conn = isolated_conn
        body = _post_body()

        def attempt(_thread_id: int):
            try:
                store(
                    conn,
                    user_id="u_alice",
                    endpoint="POST /payments",
                    key="race-key",
                    body=body,
                    response_status=201,
                    response_body="resp",
                    created_at="2026-09-24T11:04:03+00:00",
                )
            except (sqlite3.IntegrityError, sqlite3.OperationalError):
                pass

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(attempt, i) for i in range(4)]
            for f in as_completed(futures):
                pass

        rows = conn.execute(
            "SELECT COUNT(*) AS n FROM idempotency_keys "
            "WHERE user_id = ? AND endpoint = ? AND key = ?",
            ("u_alice", "POST /payments", "race-key"),
        ).fetchone()
        assert rows["n"] == 1
