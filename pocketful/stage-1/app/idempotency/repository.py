"""Idempotency store repository.

Provides ``lookup`` and ``store`` operations backed by the
``idempotency_keys`` table created in :mod:`app.idempotency.schema`.

All write paths go through the ``write_transaction`` / ``run_in_write_transaction``
wrappers from :mod:`app.sqlite_utils.transaction` so every write begins with
``BEGIN IMMEDIATE`` (spec §7 — no SELECT ... FOR UPDATE, no implicit
transactions).

The idempotency key is **stored verbatim** in its own column; the
``request_body_hash`` is a *separate derived* value computed via
:func:`app.idempotency.canonical_hash.canonical_body_hash` and stored alongside.
The two fields are never conflated.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Union

from .canonical_hash import canonical_body_hash
from ..sqlite_utils.transaction import run_in_write_transaction

__all__ = [
    "IdempotencyRecord",
    "Outcome",
    "lookup",
    "store",
    "update_response",
]


# Possible outcomes of a ``lookup`` against the store.
class Outcome:
    MISSING = "missing"   # first use of the key: caller should perform the operation
    REPLAY = "replay"     # same key + same canonical body hash: caller should return the stored response
    CONFLICT = "conflict"  # same key + different canonical body hash: caller must return 409 idempotency_key_reuse


_OUTCOMES = (Outcome.MISSING, Outcome.REPLAY, Outcome.CONFLICT)


@dataclass(frozen=True)
class IdempotencyRecord:
    """An idempotency record as stored in the database."""

    user_id: str
    endpoint: str
    key: str
    request_body_hash: str
    response_status: int
    response_body: str
    created_at: str


_BodyLike = Union[bytes, bytearray, str, dict, list, None]


def _compute_body_hash(body: _BodyLike) -> str:
    """Compute the canonical body hash for storage.

    Raises ``CanonicalHashError`` if ``body`` cannot be canonicalised.
    """
    return canonical_body_hash(body)


def _fetch(
    conn: sqlite3.Connection, user_id: str, endpoint: str, key: str
) -> IdempotencyRecord | None:
    """Read-side helper: SELECT the existing record for a (user, endpoint, key) tuple.

    Returns ``None`` when no record exists. This is a read, not a write;
    it does not require ``BEGIN IMMEDIATE`` and does not acquire a write lock.
    """
    row = conn.execute(
        "SELECT user_id, endpoint, key, request_body_hash, response_status, "
        "response_body, created_at "
        "FROM idempotency_keys "
        "WHERE user_id = ? AND endpoint = ? AND key = ?",
        (user_id, endpoint, key),
    ).fetchone()
    if row is None:
        return None
    return IdempotencyRecord(
        user_id=row["user_id"],
        endpoint=row["endpoint"],
        key=row["key"],
        request_body_hash=row["request_body_hash"],
        response_status=row["response_status"],
        response_body=row["response_body"],
        created_at=row["created_at"],
    )


def lookup(
    conn: sqlite3.Connection,
    user_id: str,
    endpoint: str,
    key: str,
    body: _BodyLike,
) -> tuple[str, IdempotencyRecord | None]:
    """Classify an incoming request against the idempotency store.

    Given the authenticated ``user_id``, the ``endpoint`` identifier (e.g.
    ``"POST /payments"``), the client-supplied ``key`` and the parsed ``body``,
    return one of:

    * ``Outcome.MISSING``  — no record exists; caller should perform the operation.
    * ``Outcome.REPLAY``   — same key + same canonical body hash; ``record`` is the
      stored row and the caller must return the stored response verbatim.
    * ``Outcome.CONFLICT`` — same key + different canonical body hash; caller must
      respond with 409 ``idempotency_key_reuse``.

    The lookup is read-only and does not require a write transaction.
    """
    record = _fetch(conn, user_id, endpoint, key)
    if record is None:
        return Outcome.MISSING, None

    body_hash = _compute_body_hash(body)
    if record.request_body_hash == body_hash:
        return Outcome.REPLAY, record
    return Outcome.CONFLICT, record


def store(
    conn: sqlite3.Connection,
    user_id: str,
    endpoint: str,
    key: str,
    body: _BodyLike,
    response_status: int,
    response_body: str,
    created_at: str,
) -> IdempotencyRecord:
    """Persist a new idempotency record.

    Writes run inside ``BEGIN IMMEDIATE`` via the write-transaction wrapper
    from :mod:`app.sqlite_utils.transaction`. On unique-constraint violation
    (the same key was inserted concurrently between the read and the write)
    the transaction rolls back and ``sqlite3.IntegrityError`` propagates so
    the caller can re-run ``lookup`` to decide between REPLAY and CONFLICT.

    The idempotency key is stored verbatim; the canonical body hash is
    computed from ``body`` and stored as a separate derived field.
    """
    body_hash = _compute_body_hash(body)

    def _insert(cur: sqlite3.Cursor) -> IdempotencyRecord:
        cur.execute(
            "INSERT INTO idempotency_keys("
            "  user_id, endpoint, key, request_body_hash,"
            "  response_status, response_body, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                endpoint,
                key,
                body_hash,
                response_status,
                response_body,
                created_at,
            ),
        )
        return IdempotencyRecord(
            user_id=user_id,
            endpoint=endpoint,
            key=key,
            request_body_hash=body_hash,
            response_status=response_status,
            response_body=response_body,
            created_at=created_at,
        )

    return run_in_write_transaction(conn, _insert)


def update_response(
    conn: sqlite3.Connection,
    user_id: str,
    endpoint: str,
    key: str,
    response_status: int,
    response_body: str,
) -> None:
    """Update the stored response for an existing record.

    Useful when the response payload is computed *after* the record has been
    claimed by ``store`` (e.g. the payment body has been generated and
    the row needs to be enriched with the JSON of the response). Like
    ``store``, this runs inside ``BEGIN IMMEDIATE``.
    """
    def _update(cur: sqlite3.Cursor) -> None:
        cur.execute(
            "UPDATE idempotency_keys "
            "SET response_status = ?, response_body = ? "
            "WHERE user_id = ? AND endpoint = ? AND key = ?",
            (response_status, response_body, user_id, endpoint, key),
        )

    run_in_write_transaction(conn, _update)