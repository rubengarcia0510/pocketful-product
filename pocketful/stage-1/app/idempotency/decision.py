"""Idempotency decision logic.

Layered on top of the store (PDF-31): this module exposes the *decision* an
HTTP handler must make for an incoming idempotent request, namely:

* ``Outcome.MISSING``  — the key has not been used by this user on this
  endpoint yet; the caller should perform the operation and persist the
  outcome.
* ``Outcome.REPLAY``   — the same key + same canonical body has been seen
  before; the caller must return the stored response verbatim (no further
  state changes).
* ``Outcome.CONFLICT`` — the same key has been used by this user on this
  endpoint, but the canonical body differs; the caller must respond with
  409 ``idempotency_key_reuse``.

The idempotency key is matched **verbatim**, scoped strictly by the
authenticated ``user_id`` and the ``endpoint`` identifier. The canonical
request-body hash is a *separate* derived field (computed via
:func:`app.idempotency.canonical_hash.canonical_body_hash`); the two values
are never conflated.

All write paths go through ``run_in_write_transaction`` from
:mod:`app.sqlite_utils.transaction` so every mutation begins with
``BEGIN IMMEDIATE`` (spec §7). ``SELECT ... FOR UPDATE`` is never used.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Union

from .canonical_hash import canonical_body_hash
from .repository import (
    IdempotencyRecord,
    Outcome,
    lookup as repo_lookup,
    store as repo_store,
    update_response as repo_update_response,
)

__all__ = [
    "Decision",
    "ClaimResult",
    "decide",
    "claim",
]


_BodyLike = Union[bytes, bytearray, str, dict, list, None]


@dataclass(frozen=True)
class Decision:
    """Result of resolving an incoming idempotent request.

    ``outcome`` is one of ``Outcome.MISSING`` / ``Outcome.REPLAY`` /
    ``Outcome.CONFLICT``.

    ``record`` is the stored row when ``outcome`` is REPLAY or CONFLICT, and
    ``None`` when ``outcome`` is MISSING.
    """

    outcome: str
    record: IdempotencyRecord | None

    @property
    def is_missing(self) -> bool:
        return self.outcome == Outcome.MISSING

    @property
    def is_replay(self) -> bool:
        return self.outcome == Outcome.REPLAY

    @property
    def is_conflict(self) -> bool:
        return self.outcome == Outcome.CONFLICT

    @property
    def response_status(self) -> int | None:
        """Convenience: HTTP status from the stored record, or ``None``."""
        if self.record is None:
            return None
        return self.record.response_status

    @property
    def response_body(self) -> str | None:
        """Convenience: stored JSON response body, or ``None``."""
        if self.record is None:
            return None
        return self.record.response_body

    @property
    def response_payload(self) -> Any | None:
        """Convenience: the stored response body parsed as JSON.

        Returns ``None`` if there is no record or the stored body is empty
        (placeholder written before the operation completed).
        """
        if self.record is None or not self.record.response_body:
            return None
        try:
            return json.loads(self.record.response_body)
        except json.JSONDecodeError:
            return None


def decide(
    conn: sqlite3.Connection,
    user_id: str,
    endpoint: str,
    key: str,
    body: _BodyLike,
) -> Decision:
    """Resolve the idempotency decision for an incoming request.

    Given the authenticated ``user_id``, the ``endpoint`` identifier (e.g.
    ``"POST /payments"``), the client-supplied ``key`` and the parsed ``body``,
    return a :class:`Decision` with one of:

    * :attr:`Outcome.MISSING`  — first use of the key for this (user, endpoint)
      pair; caller should perform the operation.
    * :attr:`Outcome.REPLAY`   — same key + same canonical body; the caller
      must return ``decision.response_status`` / ``decision.response_body``
      verbatim and **must not** perform the operation again.
    * :attr:`Outcome.CONFLICT` — same key + different canonical body; caller
      must respond with 409 ``idempotency_key_reuse``.

    Scoping is strict: a key collision across users, or across endpoints for
    the same user, never produces a REPLAY or CONFLICT.

    The decision is read-only and does not require a write transaction.
    """
    outcome, record = repo_lookup(conn, user_id, endpoint, key, body)
    return Decision(outcome=outcome, record=record)


def claim(
    conn: sqlite3.Connection,
    user_id: str,
    endpoint: str,
    key: str,
    body: _BodyLike,
    response_status: int,
    response_body: str,
    created_at: str,
) -> ClaimResult:
    """Atomically claim an idempotency key for the first use.

    The caller has already decided via :func:`decide` that the key is unused
    and is now attempting to persist a placeholder record so that a
    concurrent caller with the same key cannot also claim it.

    Behaviour:

    * **Won the race** — :attr:`ClaimResult.outcome` is ``Outcome.MISSING``
      and :attr:`ClaimResult.is_winner` is ``True``. The caller should now
      perform the operation and call :func:`update_response` with the real
      response body.
    * **Lost the race** — another caller stored a record between our
      :func:`decide` and our :func:`store`. The function re-runs :func:`decide`
      and returns the actual outcome (``REPLAY`` or ``CONFLICT``) along with
      the stored record, so the caller can return the stored response or
      409 ``idempotency_key_reuse`` without re-executing the operation.

    In both branches the ``key`` is stored verbatim and the canonical body
    hash is computed from ``body`` via :func:`canonical_body_hash`. All
    writes happen inside ``BEGIN IMMEDIATE``.
    """
    # First decide — fast path: if the key is already taken, short-circuit.
    initial = decide(conn, user_id, endpoint, key, body)
    if initial.outcome != Outcome.MISSING:
        return ClaimResult(
            outcome=initial.outcome,
            record=initial.record,
            is_winner=False,
        )

    # Try to claim the key with a placeholder response.
    try:
        record = repo_store(
            conn,
            user_id=user_id,
            endpoint=endpoint,
            key=key,
            body=body,
            response_status=response_status,
            response_body=response_body,
            created_at=created_at,
        )
        return ClaimResult(
            outcome=Outcome.MISSING,
            record=record,
            is_winner=True,
        )
    except sqlite3.IntegrityError:
        # Concurrent winner already inserted the row. Re-decide to learn
        # whether our body matches (REPLAY) or differs (CONFLICT).
        second = decide(conn, user_id, endpoint, key, body)
        return ClaimResult(
            outcome=second.outcome,
            record=second.record,
            is_winner=False,
        )


@dataclass(frozen=True)
class ClaimResult:
    """Result of a :func:`claim` attempt.

    * :attr:`outcome` is the *effective* outcome once the race has been
      resolved:

      - :attr:`Outcome.MISSING` (with ``is_winner=True``) — we won the race
        and the caller must perform the operation, then call
        :func:`update_response` with the real response.
      - :attr:`Outcome.REPLAY` — another caller won and stored a record with
        the same canonical body; return the stored response verbatim.
      - :attr:`Outcome.CONFLICT` — another caller won and stored a record
        with a different canonical body; return 409 ``idempotency_key_reuse``.

    * :attr:`record` is the stored record (always set when ``is_winner`` is
      ``False`` and the race was lost; ``None`` when ``is_winner`` is
      ``True`` — the caller has not yet computed the real response).
    """

    outcome: str
    record: IdempotencyRecord | None
    is_winner: bool

    @property
    def is_replay(self) -> bool:
        return self.outcome == Outcome.REPLAY

    @property
    def is_conflict(self) -> bool:
        return self.outcome == Outcome.CONFLICT

    @property
    def response_status(self) -> int | None:
        if self.record is None:
            return None
        return self.record.response_status

    @property
    def response_body(self) -> str | None:
        if self.record is None:
            return None
        return self.record.response_body

    @property
    def response_payload(self) -> Any | None:
        """Convenience: stored response body parsed as JSON, or ``None``."""
        if self.record is None or not self.record.response_body:
            return None
        try:
            return json.loads(self.record.response_body)
        except json.JSONDecodeError:
            return None


def update_response(
    conn: sqlite3.Connection,
    user_id: str,
    endpoint: str,
    key: str,
    response_status: int,
    response_body: str,
) -> None:
    """Update the stored response for a record we previously claimed.

    Thin re-export of :func:`app.idempotency.repository.update_response` so
    callers only need to import from this module. The write runs inside
    ``BEGIN IMMEDIATE``.
    """
    repo_update_response(
        conn,
        user_id=user_id,
        endpoint=endpoint,
        key=key,
        response_status=response_status,
        response_body=response_body,
    )