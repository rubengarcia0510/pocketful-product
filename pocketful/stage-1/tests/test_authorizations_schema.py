"""Schema v3 migration tests (PDF-35).

Stage-2 introduces the ``authorizations`` and ``authorization_captures``
tables plus a nullable ``authorization_id`` column on ``payments``. The
migration must apply cleanly on a brand-new database **and** on a
database that was already at schema v2 carrying real stage-1 data
(users, payments, payment requests, splits, idempotency keys).

The v1→v2 migration is verified elsewhere (PDF-31). The v2→v3 path
that we ship here must not lose any v2 data and must be safe to run
multiple times.
"""
from __future__ import annotations

import sqlite3
from typing import Iterator

import pytest

from app import db as db_mod
from app.authorizations.schema import apply_schema as apply_v3_schema
from app.sqlite_utils.transaction import write_transaction


@pytest.fixture
def v2_conn(tmp_path) -> Iterator[sqlite3.Connection]:
    """A fresh SQLite database at schema version 2 with stage-1 data seeded.

    The fixture deliberately does not call :func:`db.init_schema`; that
    would advance all the way to the current schema version (3) and
    create the v3 tables, defeating the point of the migration test.
    Instead we call the v1/v2 appliers directly and stamp
    ``user_version`` at ``2`` to simulate a real pre-stage-2 database.
    """
    path = str(tmp_path / "stage1.db")
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("BEGIN IMMEDIATE")
    try:
        db_mod._apply_v1(conn)
        db_mod._apply_v2(conn)
        conn.execute("PRAGMA user_version = 2")
    finally:
        conn.execute("COMMIT")
    _seed_stage1_state(conn)
    yield conn
    conn.close()


def _seed_stage1_state(conn: sqlite3.Connection) -> None:
    """Insert a representative slice of stage-1 state for migration tests."""
    now = "2026-01-01T00:00:00+00:00"
    with write_transaction(conn) as cur:
        cur.execute(
            "INSERT INTO service_meta(key, value) VALUES('currency', 'EUR')"
        )
        cur.execute(
            "INSERT INTO service_meta(key, value) VALUES('minor_units', '2')"
        )
        cur.execute(
            "INSERT INTO users(id, email, password_hash, display_name, "
            "handle, balance, currency, minor_units, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "u_ada",
                "ada@example.com",
                "hash",
                "Ada",
                "ada",
                10000,
                "EUR",
                2,
                now,
            ),
        )
        cur.execute(
            "INSERT INTO users(id, email, password_hash, display_name, "
            "handle, balance, currency, minor_units, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "u_bob",
                "bob@example.com",
                "hash",
                "Bob",
                "bob",
                2500,
                "EUR",
                2,
                now,
            ),
        )
        cur.execute(
            "INSERT INTO payments(id, from_user_id, to_user_id, amount, "
            "note, visibility, request_id, settlement_id, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "p_1",
                "u_ada",
                "u_bob",
                500,
                "coffee",
                "public",
                None,
                None,
                now,
            ),
        )
        cur.execute(
            "INSERT INTO payment_requests(id, requester_id, payer_id, "
            "amount, note, status, payment_id, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "rq_1",
                "u_bob",
                "u_ada",
                1200,
                "taxi",
                "pending",
                None,
                now,
            ),
        )
        cur.execute(
            "INSERT INTO idempotency_keys(user_id, endpoint, key, "
            "request_body_hash, response_status, response_body, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?)",
            (
                "u_ada",
                "POST /payments",
                "key-1",
                "deadbeef",
                201,
                "{}",
                now,
            ),
        )


def _user_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0])


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','index')"
    ).fetchall()
    return {row["name"] for row in rows}


def _payments_columns(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("PRAGMA table_info(payments)").fetchall()
    return {row["name"] for row in rows}


def test_v3_migration_advances_user_version(v2_conn):
    """Applying the v3 path raises user_version from 2 to 3."""
    assert _user_version(v2_conn) == 2
    apply_v3_schema(v2_conn)
    assert _user_version(v2_conn) == 3


def test_v3_migration_adds_new_tables_and_indexes(v2_conn):
    """After the migration the new tables and their indexes exist."""
    apply_v3_schema(v2_conn)
    names = _table_names(v2_conn)
    for required in {
        "authorizations",
        "authorization_captures",
        "idx_authorizations_from",
        "idx_authorizations_to",
        "idx_authorizations_status",
        "idx_authorizations_expires",
        "idx_auth_captures_payment",
        "idx_payments_authorization",
    }:
        assert required in names, f"missing schema object: {required}"


def test_v3_migration_adds_authorization_id_to_payments(v2_conn):
    """The new ``payments.authorization_id`` column is nullable and indexed."""
    before = _payments_columns(v2_conn)
    assert "authorization_id" not in before
    apply_v3_schema(v2_conn)
    cols = _payments_columns(v2_conn)
    assert "authorization_id" in cols
    # NULL by default (column added without DEFAULT NOT NULL).
    row = v2_conn.execute(
        "SELECT authorization_id FROM payments WHERE id = ?", ("p_1",)
    ).fetchone()
    assert row["authorization_id"] is None


def test_v3_migration_preserves_stage1_data(v2_conn):
    """Every row that existed at v2 still exists at v3."""
    apply_v3_schema(v2_conn)

    users = v2_conn.execute("SELECT id, balance FROM users ORDER BY id").fetchall()
    assert [(u["id"], u["balance"]) for u in users] == [
        ("u_ada", 10000),
        ("u_bob", 2500),
    ]

    payment = v2_conn.execute(
        "SELECT id, amount, note, visibility FROM payments WHERE id = 'p_1'"
    ).fetchone()
    assert payment["amount"] == 500
    assert payment["note"] == "coffee"
    assert payment["visibility"] == "public"

    request = v2_conn.execute(
        "SELECT id, amount, status FROM payment_requests WHERE id = 'rq_1'"
    ).fetchone()
    assert request["amount"] == 1200
    assert request["status"] == "pending"

    idem = v2_conn.execute(
        "SELECT user_id, endpoint, key FROM idempotency_keys"
    ).fetchone()
    assert (idem["user_id"], idem["endpoint"], idem["key"]) == (
        "u_ada",
        "POST /payments",
        "key-1",
    )


def test_v3_migration_is_idempotent(v2_conn):
    """Running the v3 DDL twice does not error or duplicate objects."""
    apply_v3_schema(v2_conn)
    apply_v3_schema(v2_conn)
    assert _user_version(v2_conn) == 3
    # One row in payments; the second pass must not have duplicated the
    # ALTER TABLE column. SQLite raises "duplicate column" if we miss the
    # existence check, so a clean second pass is the proof.
    count = v2_conn.execute("SELECT COUNT(*) AS c FROM payments").fetchone()["c"]
    assert count == 1


def test_v3_migration_keeps_invariants(v2_conn):
    """Sum of balances still equals the seeded total after the v3 path."""
    apply_v3_schema(v2_conn)
    total = v2_conn.execute("SELECT SUM(balance) AS s FROM users").fetchone()["s"]
    assert total == 12500


def test_init_schema_on_fresh_db_lands_at_v3(tmp_path):
    """The shipped :func:`db.init_schema` brings a new DB to user_version 3."""
    path = str(tmp_path / "fresh.db")
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    db_mod.init_schema(conn)
    cur = conn.execute("PRAGMA user_version").fetchone()
    assert int(cur[0]) == 3
    # All v3 objects exist.
    names = _table_names(conn)
    for required in {
        "authorizations",
        "authorization_captures",
        "idempotency_keys",
        "split_participants",
        "payments",
        "users",
    }:
        assert required in names
    # authorization_id is present on payments.
    assert "authorization_id" in _payments_columns(conn)
    conn.close()
