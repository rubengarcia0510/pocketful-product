"""Schema (DDL) for the authorisations subsystem (stage-2, schema v3).

The DDL is published here in one place so it can be applied both by
``app.db._apply_v3`` (on first boot / migration) and by tests that
need to rebuild the schema on an isolated connection. Keep the
definitions byte-for-byte identical to the inline DDL emitted by
``_apply_v3``; the inline path is the source of truth at startup and
this module is the source of truth for anything that needs the
statements outside of ``init_schema``.
"""
from __future__ import annotations

import sqlite3

__all__ = [
    "AUTHORIZATIONS_TABLE",
    "AUTHORIZATIONS_INDEXES",
    "AUTHORIZATION_CAPTURES_TABLE",
    "AUTHORIZATION_CAPTURES_INDEXES",
    "PAYMENT_AUTHORIZATION_COLUMN_SQL",
    "SCHEMA_VERSION",
    "apply_schema",
]


SCHEMA_VERSION = 3


AUTHORIZATIONS_TABLE = (
    "CREATE TABLE IF NOT EXISTS authorizations ("
    "  id TEXT PRIMARY KEY,"
    "  from_user_id TEXT NOT NULL REFERENCES users(id),"
    "  to_user_id TEXT NOT NULL REFERENCES users(id),"
    "  amount INTEGER NOT NULL,"
    "  captured_amount INTEGER NOT NULL DEFAULT 0,"
    "  note TEXT NOT NULL DEFAULT '',"
    "  visibility TEXT NOT NULL DEFAULT 'public',"
    "  status TEXT NOT NULL CHECK (status IN ('open', 'captured', 'voided', 'expired')),"
    "  expires_at TEXT NOT NULL,"
    "  latest_payment_id TEXT REFERENCES payments(id),"
    "  created_at TEXT NOT NULL,"
    "  CHECK (amount >= 1),"
    "  CHECK (captured_amount >= 0),"
    "  CHECK (captured_amount <= amount)"
    ")"
)

_AUTHORIZATIONS_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_authorizations_from ON authorizations(from_user_id)",
    "CREATE INDEX IF NOT EXISTS idx_authorizations_to ON authorizations(to_user_id)",
    "CREATE INDEX IF NOT EXISTS idx_authorizations_status ON authorizations(status)",
    "CREATE INDEX IF NOT EXISTS idx_authorizations_expires ON authorizations(expires_at)",
)
AUTHORIZATIONS_INDEXES = _AUTHORIZATIONS_INDEXES


AUTHORIZATION_CAPTURES_TABLE = (
    "CREATE TABLE IF NOT EXISTS authorization_captures ("
    "  authorization_id TEXT NOT NULL REFERENCES authorizations(id) ON DELETE CASCADE,"
    "  seq INTEGER NOT NULL,"
    "  payment_id TEXT NOT NULL REFERENCES payments(id),"
    "  amount INTEGER NOT NULL,"
    "  created_at TEXT NOT NULL,"
    "  PRIMARY KEY (authorization_id, seq),"
    "  CHECK (seq >= 1),"
    "  CHECK (amount >= 1)"
    ")"
)

_AUTHORIZATION_CAPTURES_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_auth_captures_payment ON authorization_captures(payment_id)",
)
AUTHORIZATION_CAPTURES_INDEXES = _AUTHORIZATION_CAPTURES_INDEXES


PAYMENT_AUTHORIZATION_COLUMN_SQL = (
    "ALTER TABLE payments ADD COLUMN authorization_id TEXT REFERENCES authorizations(id)"
)


def apply_schema(conn: sqlite3.Connection) -> None:
    """Apply the stage-2 authorisations DDL to ``conn``.

    Safe to call repeatedly; ``CREATE TABLE / INDEX IF NOT EXISTS`` are
    no-ops when the objects already exist. The ``ALTER TABLE payments``
    statement is guarded by a column-existence check and so is also
    idempotent across repeated calls.

    After the DDL is applied, ``PRAGMA user_version`` is advanced to
    :data:`SCHEMA_VERSION` (``3``). This makes the function a complete
    "apply v3 schema" operation suitable for both production migrations
    and isolated test connections; the production
    :func:`app.db.init_schema` orchestrator will re-stamp the same
    value at the end of its own migration transaction, which is a no-op.

    The function performs no implicit transaction; the caller is
    responsible for wrapping it in the BEGIN IMMEDIATE write-transaction
    wrapper from :mod:`app.sqlite_utils.transaction` when running
    alongside other writes.
    """
    conn.execute(AUTHORIZATIONS_TABLE)
    for stmt in _AUTHORIZATIONS_INDEXES:
        conn.execute(stmt)
    conn.execute(AUTHORIZATION_CAPTURES_TABLE)
    for stmt in _AUTHORIZATION_CAPTURES_INDEXES:
        conn.execute(stmt)
    rows = conn.execute("PRAGMA table_info(payments)").fetchall()
    if not any(row["name"] == "authorization_id" for row in rows):
        conn.execute(PAYMENT_AUTHORIZATION_COLUMN_SQL)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_payments_authorization ON payments(authorization_id)"
    )
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
