"""Schema (DDL) for the idempotency store.

Per spec §7 (stage-1.md), idempotency keys are scoped to (user, endpoint, key)
so two different users may use the same key string with no interaction between
them, and the same key on a different endpoint is a distinct record.

The key is stored verbatim as its own column (``key``) — never replaced by
its hash. The canonical request-body hash is stored alongside as a separate
derived column (``request_body_hash``) computed via
``app.idempotency.canonical_hash.canonical_body_hash``.
"""

from __future__ import annotations

import sqlite3

__all__ = ["IDEMPOTENCY_TABLE", "apply_schema"]


# Statement that (idempotently) creates the table and its supporting index.
#
# The composite PRIMARY KEY on (user_id, endpoint, key) enforces the spec
# scoping at the storage layer; two distinct users reusing the same key string
# do not collide, and the same key on a different endpoint is a separate row.
IDEMPOTENCY_TABLE = (
    "CREATE TABLE IF NOT EXISTS idempotency_keys ("
    "  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,"
    "  endpoint TEXT NOT NULL,"
    "  key TEXT NOT NULL,"
    "  request_body_hash TEXT NOT NULL,"
    "  response_status INTEGER NOT NULL,"
    "  response_body TEXT NOT NULL,"
    "  created_at TEXT NOT NULL,"
    "  PRIMARY KEY (user_id, endpoint, key)"
    ")"
)


_IDEMPOTENCY_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_idem_user ON idempotency_keys(user_id)"
)


def apply_schema(conn: sqlite3.Connection) -> None:
    """Apply the idempotency store DDL to the given connection.

    This is safe to call multiple times: ``CREATE TABLE IF NOT EXISTS`` and
    ``CREATE INDEX IF NOT EXISTS`` are no-ops when the objects already exist.

    The function performs no implicit transaction; the caller is responsible
    for wrapping it in ``write_transaction`` when running through the
    BEGIN IMMEDIATE wrapper from ``app.sqlite_utils.transaction``.
    """
    conn.execute(IDEMPOTENCY_TABLE)
    conn.execute(_IDEMPOTENCY_INDEX)