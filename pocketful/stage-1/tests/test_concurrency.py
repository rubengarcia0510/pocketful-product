"""Concurrency tests for idempotent writes (PDF-34).

These tests stress ``claim()`` from
:mod:`app.idempotency.decision` under concurrent load against the
SQLite store. They verify the contract:

* When N threads race for the same ``(user, endpoint, key)`` tuple with
  the **same canonical body**, exactly one wins (``Outcome.MISSING`` /
  ``is_winner=True``) and every other thread sees ``Outcome.REPLAY``
  pointing at the winner's record. Exactly **one** row persists.
* When N threads race for the same tuple with **distinct canonical
  bodies**, exactly one wins and every other thread sees
  ``Outcome.CONFLICT``. Exactly **one** row persists (the winner's).
* After sustained concurrent load the store remains consistent:
  ``PRAGMA integrity_check`` returns ``ok``, and per-tuple row counts
  never exceed 1.
* When N threads each target a **different** key, every thread wins
  (all ``MISSING``), and the store contains exactly N rows, one per key.

The implementation guarantees serialisation through
``BEGIN IMMEDIATE`` (PDF-30) and the composite primary key
``(user_id, endpoint, key)`` on ``idempotency_keys`` (PDF-31), so the
race window is the one between ``decide()`` and the ``INSERT`` inside
``store()``. The losing threads re-run ``decide()`` after catching the
``IntegrityError`` and learn the actual outcome from the winner's row.

Each thread opens its own ``sqlite3.Connection`` against the per-test
``tmp_path``-backed file: sharing a single ``Connection`` across
threads is unsafe regardless of ``check_same_thread=False``.
"""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pytest

from app.idempotency.canonical_hash import canonical_body_hash
from app.idempotency.decision import ClaimResult, claim
from app.idempotency.repository import Outcome
from app.sqlite_utils.transaction import write_transaction


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_db_path(tmp_path):
    """Provide an isolated DB file path with schema applied.

    Each thread is expected to open its own SQLite connection against
    this path; sharing one ``Connection`` across threads is unsafe.
    The file is removed at the end of the test by pytest's ``tmp_path``.
    """
    path = str(tmp_path / "concurrent.db")

    setup_conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    setup_conn.row_factory = sqlite3.Row
    setup_conn.execute("PRAGMA journal_mode = WAL")
    setup_conn.execute("PRAGMA busy_timeout = 5000")

    setup_conn.execute(
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

    with write_transaction(setup_conn) as cur:
        cur.execute(
            "INSERT INTO users(id, email, password_hash, display_name, handle, "
            "balance, currency, minor_units, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("u_alice", "a@x", "x", "Alice", "alice", 0, "EUR", 2,
             "2026-01-01T00:00:00+00:00"),
        )

    from app.idempotency.schema import apply_schema
    apply_schema(setup_conn)
    setup_conn.close()

    yield path


def _open_thread_conn(db_path: str) -> sqlite3.Connection:
    """Open a per-thread SQLite connection (WAL + busy_timeout)."""
    conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _payment_body(amount: int = 1500, note: str = "dinner") -> dict:
    return {"to_handle": "bob", "amount": amount, "note": note, "visibility": "public"}


def _check_integrity(db_path: str) -> str:
    """Run ``PRAGMA integrity_check`` and return the result string.

    A healthy store returns the single row ``"ok"``.
    """
    conn = _open_thread_conn(db_path)
    try:
        rows = conn.execute("PRAGMA integrity_check").fetchall()
        return " ".join(r[0] for r in rows)
    finally:
        conn.close()


def _count_rows(db_path: str, user_id: str, endpoint: str, key: str) -> int:
    """Count rows in ``idempotency_keys`` for the given tuple."""
    conn = _open_thread_conn(db_path)
    try:
        return conn.execute(
            "SELECT COUNT(*) AS n FROM idempotency_keys "
            "WHERE user_id = ? AND endpoint = ? AND key = ?",
            (user_id, endpoint, key),
        ).fetchone()["n"]
    finally:
        conn.close()


def _row_for(db_path: str, user_id: str, endpoint: str, key: str) -> dict | None:
    """Return the stored row for the tuple, or ``None`` if missing."""
    conn = _open_thread_conn(db_path)
    try:
        return conn.execute(
            "SELECT user_id, endpoint, key, request_body_hash, response_status, "
            "response_body, created_at FROM idempotency_keys "
            "WHERE user_id = ? AND endpoint = ? AND key = ?",
            (user_id, endpoint, key),
        ).fetchone()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Same body, same key: 1 winner + N-1 REPLAY
# ---------------------------------------------------------------------------


class TestConcurrentClaimSameBody:
    """N threads racing on the same tuple with identical canonical bodies."""

    def test_eight_threads_same_body_one_winner_seven_replay(self, isolated_db_path):
        db_path = isolated_db_path
        body = _payment_body()
        key = "concurrency-same-body-8"
        num_threads = 8
        barrier = threading.Barrier(num_threads)

        def attempt(_thread_id: int) -> ClaimResult:
            c = _open_thread_conn(db_path)
            try:
                barrier.wait()
                return claim(
                    c,
                    user_id="u_alice",
                    endpoint="POST /payments",
                    key=key,
                    body=body,
                    response_status=201,
                    response_body="",
                    created_at="2026-10-05T00:00:00+00:00",
                )
            finally:
                c.close()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(attempt, i) for i in range(num_threads)]
            results = [f.result() for f in as_completed(futures)]

        winners = [r for r in results if r.is_winner]
        losers = [r for r in results if not r.is_winner]

        assert len(winners) == 1, f"expected 1 winner, got {len(winners)}"
        assert len(losers) == num_threads - 1

        winner = winners[0]
        assert winner.outcome == Outcome.MISSING
        assert winner.record is not None
        assert winner.record.user_id == "u_alice"
        assert winner.record.key == key
        assert winner.record.request_body_hash == canonical_body_hash(body)

        for loser in losers:
            assert loser.outcome == Outcome.REPLAY, (
                f"same body must yield REPLAY, got {loser.outcome}"
            )
            assert loser.is_replay
            assert not loser.is_conflict
            assert loser.record is not None
            assert loser.record.user_id == "u_alice"
            assert loser.record.key == key
            # Loser sees the winner's record, not a synthesised one.
            assert loser.record.request_body_hash == winner.record.request_body_hash
            assert loser.record.response_body == winner.record.response_body

        # Exactly one row persisted for the contested tuple.
        assert _count_rows(db_path, "u_alice", "POST /payments", key) == 1

        # The store must still be consistent.
        assert _check_integrity(db_path) == "ok"

    def test_sixteen_threads_same_body_single_row(self, isolated_db_path):
        """Stress test: 16 threads, identical bodies — never more than 1 row."""
        db_path = isolated_db_path
        body = _payment_body()
        key = "concurrency-same-body-16"
        num_threads = 16
        barrier = threading.Barrier(num_threads)

        def attempt(_thread_id: int) -> ClaimResult:
            c = _open_thread_conn(db_path)
            try:
                barrier.wait()
                return claim(
                    c,
                    user_id="u_alice",
                    endpoint="POST /payments",
                    key=key,
                    body=body,
                    response_status=201,
                    response_body="",
                    created_at="2026-10-05T00:00:00+00:00",
                )
            finally:
                c.close()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(attempt, i) for i in range(num_threads)]
            results = [f.result() for f in as_completed(futures)]

        winners = [r for r in results if r.is_winner]
        losers = [r for r in results if not r.is_winner]

        assert len(winners) == 1
        assert len(losers) == num_threads - 1
        for loser in losers:
            assert loser.outcome == Outcome.REPLAY

        assert _count_rows(db_path, "u_alice", "POST /payments", key) == 1
        assert _check_integrity(db_path) == "ok"


# ---------------------------------------------------------------------------
# Different bodies, same key: 1 winner + N-1 CONFLICT
# ---------------------------------------------------------------------------


class TestConcurrentClaimDifferentBodies:
    """N threads racing on the same tuple with distinct canonical bodies."""

    def test_eight_threads_distinct_bodies_one_winner_seven_conflict(
        self, isolated_db_path
    ):
        db_path = isolated_db_path
        key = "concurrency-distinct-bodies-8"

        # Each thread sends a body with a unique ``amount`` so the canonical
        # hashes differ. The barrier ensures all threads pass ``decide()``
        # (which returns MISSING for all of them) before any of them gets
        # to the serialised ``INSERT``.
        num_threads = 8
        bodies = [_payment_body(amount=1000 + i) for i in range(num_threads)]
        barrier = threading.Barrier(num_threads)

        def attempt(thread_id: int) -> ClaimResult:
            c = _open_thread_conn(db_path)
            try:
                barrier.wait()
                return claim(
                    c,
                    user_id="u_alice",
                    endpoint="POST /payments",
                    key=key,
                    body=bodies[thread_id],
                    response_status=201,
                    response_body="",
                    created_at="2026-10-05T00:00:00+00:00",
                )
            finally:
                c.close()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(attempt, i) for i in range(num_threads)]
            results = [f.result() for f in as_completed(futures)]

        winners = [r for r in results if r.is_winner]
        losers = [r for r in results if not r.is_winner]

        assert len(winners) == 1
        assert len(losers) == num_threads - 1

        winner = winners[0]
        assert winner.outcome == Outcome.MISSING
        assert winner.record is not None
        winner_hash = winner.record.request_body_hash

        # Every loser must observe CONFLICT: the winner stored *some* body,
        # and that body differs from every other body's canonical hash.
        for loser in losers:
            assert loser.outcome == Outcome.CONFLICT, (
                f"distinct bodies must yield CONFLICT, got {loser.outcome}"
            )
            assert loser.is_conflict
            assert not loser.is_replay
            assert loser.record is not None
            # The loser sees the winner's record (same hash).
            assert loser.record.request_body_hash == winner_hash

        # Exactly one row persisted: the winner's body. No duplicates.
        assert _count_rows(db_path, "u_alice", "POST /payments", key) == 1
        # The stored hash matches the winner's body.
        stored = _row_for(db_path, "u_alice", "POST /payments", key)
        assert stored is not None
        assert stored["request_body_hash"] == winner_hash

        assert _check_integrity(db_path) == "ok"

    def test_two_threads_different_bodies_one_winner_one_conflict(
        self, isolated_db_path
    ):
        """Minimum-size race: 2 threads, different bodies → 1 winner + 1 CONFLICT."""
        db_path = isolated_db_path
        key = "concurrency-distinct-bodies-2"
        body_a = _payment_body(amount=1000)
        body_b = _payment_body(amount=9999)
        barrier = threading.Barrier(2)

        def attempt(body: dict) -> ClaimResult:
            c = _open_thread_conn(db_path)
            try:
                barrier.wait()
                return claim(
                    c,
                    user_id="u_alice",
                    endpoint="POST /payments",
                    key=key,
                    body=body,
                    response_status=201,
                    response_body="",
                    created_at="2026-10-05T00:00:00+00:00",
                )
            finally:
                c.close()

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(attempt, body_a),
                executor.submit(attempt, body_b),
            ]
            results = [f.result() for f in as_completed(futures)]

        winners = [r for r in results if r.is_winner]
        losers = [r for r in results if not r.is_winner]

        assert len(winners) == 1
        assert len(losers) == 1
        assert winners[0].outcome == Outcome.MISSING
        assert losers[0].outcome == Outcome.CONFLICT
        assert _count_rows(db_path, "u_alice", "POST /payments", key) == 1
        assert _check_integrity(db_path) == "ok"


# ---------------------------------------------------------------------------
# No duplicate writes; store integrity under load
# ---------------------------------------------------------------------------


class TestNoDuplicateWrites:
    """Verify the store never produces duplicate rows under concurrent claim."""

    def test_repeated_contested_tuples_no_leak(self, isolated_db_path):
        """Run the contested-tuple race many times against the same key.

        Each round re-issues ``claim()`` concurrently; the resulting row
        count for the tuple must stay at 1 across all rounds, and no
        row may be created with a body hash different from the winner's.
        """
        db_path = isolated_db_path
        body = _payment_body()
        key = "concurrency-no-leak"
        rounds = 5

        for _ in range(rounds):
            num_threads = 4
            barrier = threading.Barrier(num_threads)

            def attempt(_thread_id: int) -> ClaimResult:
                c = _open_thread_conn(db_path)
                try:
                    barrier.wait()
                    return claim(
                        c,
                        user_id="u_alice",
                        endpoint="POST /payments",
                        key=key,
                        body=body,
                        response_status=201,
                        response_body="",
                        created_at="2026-10-05T00:00:00+00:00",
                    )
                finally:
                    c.close()

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(attempt, i) for i in range(num_threads)]
                _ = [f.result() for f in as_completed(futures)]

            n = _count_rows(db_path, "u_alice", "POST /payments", key)
            assert n == 1, f"round leaked duplicate rows: {n}"

        assert _check_integrity(db_path) == "ok"

    def test_distinct_keys_each_claim_once(self, isolated_db_path):
        """N threads each using a *different* key all win (no contention).

        The store must contain exactly N rows afterwards, one per key,
        and the integrity check must pass.
        """
        db_path = isolated_db_path
        num_threads = 10
        barrier = threading.Barrier(num_threads)

        def attempt(thread_id: int) -> ClaimResult:
            c = _open_thread_conn(db_path)
            try:
                barrier.wait()
                return claim(
                    c,
                    user_id="u_alice",
                    endpoint="POST /payments",
                    key=f"distinct-key-{thread_id}",
                    body=_payment_body(amount=1000 + thread_id),
                    response_status=201,
                    response_body="",
                    created_at="2026-10-05T00:00:00+00:00",
                )
            finally:
                c.close()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(attempt, i) for i in range(num_threads)]
            results = [f.result() for f in as_completed(futures)]

        # Every thread wins: distinct keys never collide.
        winners = [r for r in results if r.is_winner]
        losers = [r for r in results if not r.is_winner]
        assert len(winners) == num_threads
        assert len(losers) == 0
        for w in winners:
            assert w.outcome == Outcome.MISSING
            assert w.record is not None

        # Exactly N rows persisted, one per key, and every key is present.
        for thread_id in range(num_threads):
            n = _count_rows(
                db_path, "u_alice", "POST /payments", f"distinct-key-{thread_id}"
            )
            assert n == 1, f"key {thread_id} not present or duplicated: {n}"

        # Total row count across all keys must equal num_threads — no leaks.
        verify_conn = _open_thread_conn(db_path)
        try:
            total = verify_conn.execute(
                "SELECT COUNT(*) AS n FROM idempotency_keys"
            ).fetchone()["n"]
        finally:
            verify_conn.close()
        assert total == num_threads

        assert _check_integrity(db_path) == "ok"


class TestMixedLoadIntegrity:
    """Mixed concurrent load: different keys, contested keys, integrity check."""

    def test_mixed_contested_and_unique_keys_no_corruption(self, isolated_db_path):
        """Combine several contested tuples and several unique keys in one run.

        After the storm the store must contain exactly one row per
        tuple, ``integrity_check`` must pass, and every stored
        ``request_body_hash`` must equal the canonical hash of *some*
        body that was sent (no fabricated rows).
        """
        db_path = isolated_db_path

        contested_key = "storm-contested"
        unique_count = 6
        unique_keys = [f"storm-unique-{i}" for i in range(unique_count)]
        contested_body = _payment_body(amount=1500)

        # Map key → body that should be canonicalised for it.
        expected_hashes: dict[str, str] = {contested_key: canonical_body_hash(contested_body)}
        for i, k in enumerate(unique_keys):
            expected_hashes[k] = canonical_body_hash(_payment_body(amount=2000 + i))

        num_threads = 6 + 4  # 6 unique + 4 racing on the contested key
        barrier = threading.Barrier(num_threads)

        def attempt(thread_id: int) -> ClaimResult:
            c = _open_thread_conn(db_path)
            try:
                barrier.wait()
                if thread_id < 4:
                    # Race on the contested key, identical body.
                    return claim(
                        c,
                        user_id="u_alice",
                        endpoint="POST /payments",
                        key=contested_key,
                        body=contested_body,
                        response_status=201,
                        response_body="",
                        created_at="2026-10-05T00:00:00+00:00",
                    )
                idx = thread_id - 4
                # Unique keys, each used by one thread.
                return claim(
                    c,
                    user_id="u_alice",
                    endpoint="POST /payments",
                    key=unique_keys[idx],
                    body=_payment_body(amount=2000 + idx),
                    response_status=201,
                    response_body="",
                    created_at="2026-10-05T00:00:00+00:00",
                )
            finally:
                c.close()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(attempt, i) for i in range(num_threads)]
            results = [f.result() for f in as_completed(futures)]

        # Outcome invariants:
        # - 4 contested threads ⇒ 1 winner (MISSING) + 3 losers (REPLAY).
        # - 6 unique threads ⇒ all winners (MISSING).
        winners = [r for r in results if r.is_winner]
        losers = [r for r in results if not r.is_winner]
        assert len(winners) == unique_count + 1  # 6 unique + 1 contested winner
        assert len(losers) == 3  # 3 REPLAY losers

        for loser in losers:
            assert loser.outcome == Outcome.REPLAY
            assert loser.record is not None
            assert loser.record.key == contested_key

        # Per-tuple row counts.
        for k in [contested_key, *unique_keys]:
            n = _count_rows(db_path, "u_alice", "POST /payments", k)
            assert n == 1, f"tuple {k} has {n} rows"

        # Total row count.
        verify_conn = _open_thread_conn(db_path)
        try:
            rows: list[Any] = verify_conn.execute(
                "SELECT key, request_body_hash FROM idempotency_keys"
            ).fetchall()
        finally:
            verify_conn.close()
        assert len(rows) == unique_count + 1

        # Every row's hash equals the canonical hash of *some* body that
        # was sent — no row was fabricated with an unrelated hash.
        stored_hashes = {r["key"]: r["request_body_hash"] for r in rows}
        for key, expected in expected_hashes.items():
            assert stored_hashes[key] == expected, (
                f"key {key} stored hash {stored_hashes[key]} != expected {expected}"
            )

        # Store is healthy.
        assert _check_integrity(db_path) == "ok"
