"""Stage-2 authorisations: create hold and capture.

Wires the new idempotency infrastructure (PDF-29 / PDF-31 / PDF-32) into
the two stage-2 write endpoints defined in the spec:

* ``POST /authorizations``            — opens a hold (no money moves).
* ``POST /authorizations/{id}/captures`` — captures against an open hold
  (one or more, optionally partial).

The endpoint names follow the spec verbatim (``/captures`` plural).
Both endpoints require the ``Idempotency-Key`` header. Replay and
conflict semantics match stage-1's ``POST /payments`` envelope:

* same key + same canonical body  → replay the stored original response;
* same key + different canonical body → ``409 idempotency_key_reuse``.

The read-only ``decision.decide`` check happens BEFORE any business
validation or DB write, mirroring the stage-1 inline pattern in
:mod:`app.services.payment_service` and
:mod:`app.services.request_service`. Business state and the
``idempotency_keys`` row are persisted inside a single ``BEGIN
IMMEDIATE`` transaction (no SELECT ... FOR UPDATE, no nested
transactions) via the ``write_transaction`` context manager from
:mod:`app.sqlite_utils.transaction`. The final response body is then
written into the idempotency record via
:func:`app.idempotency.decision.update_response` in a second write
transaction.

This module deliberately does NOT replace the inline idempotency in
``payment_service`` / ``request_service``; that work is out of scope
for PDF-36.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from .. import db as db_mod
from ..authorizations import repository as authz_repo
from ..sqlite_utils.transaction import write_transaction
from ..authorizations.repository import AuthorizationNotFound
from ..errors import raise_error
from ..idempotency.canonical_hash import canonical_body_hash
from ..idempotency.decision import (
    decide,
    update_response as idempotency_update_response,
)
from ..models.balance import AmountError, validate_amount

CREATE_ENDPOINT = "POST /authorizations"
CAPTURE_ENDPOINT = "POST /authorizations/{id}/captures"

MAX_NOTE_LEN = 200
ALLOWED_VISIBILITY = frozenset({"public", "private"})
DEFAULT_TTL_SECONDS = 600


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat(timespec="seconds")


def _ttl_seconds(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT value FROM service_meta WHERE key = 'authorization_ttl_seconds'"
    ).fetchone()
    if row is None:
        return DEFAULT_TTL_SECONDS
    try:
        return int(row["value"])
    except (TypeError, ValueError):
        return DEFAULT_TTL_SECONDS


def _currency(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT value FROM service_meta WHERE key = 'currency'"
    ).fetchone()
    if row is None:
        return "EUR"
    return str(row["value"])


def _new_authorization_id() -> str:
    return f"a_{secrets.token_hex(8)}"


def _new_payment_id() -> str:
    return f"p_{secrets.token_hex(8)}"


def _compute_expires_at(now: datetime, ttl_seconds: int) -> str:
    expires = now + timedelta(seconds=ttl_seconds)
    return expires.isoformat(timespec="seconds")


def _build_create_response(
    record: authz_repo.AuthorizationRecord,
    *,
    from_handle: str,
    to_handle: str,
    currency: str,
    expires_at: str,
    captures: list[authz_repo.AuthorizationCaptureRecord],
) -> dict:
    fields = authz_repo.presentation_fields(record, captures, now_iso=_utc_now_iso())
    return {
        "authorization_id": record.id,
        "from_user_id": record.from_user_id,
        "from_handle": from_handle,
        "to_user_id": record.to_user_id,
        "to_handle": to_handle,
        "amount": record.amount,
        "captured_amount": fields["captured_amount"],
        "currency": currency,
        "note": record.note,
        "visibility": record.visibility,
        "status": fields["status"],
        "expires_at": expires_at,
        "payment_id": fields["payment_id"],
        "payment_ids": fields["payment_ids"],
        "remaining_amount": fields["remaining_amount"],
        "created_at": record.created_at,
    }


def _build_payment_response(
    payment_row: sqlite3.Row,
    *,
    from_handle: str,
    to_handle: str,
    currency: str,
    authorization_id: str,
) -> dict:
    return {
        "payment_id": payment_row["id"],
        "from_user_id": payment_row["from_user_id"],
        "from_handle": from_handle,
        "to_user_id": payment_row["to_user_id"],
        "to_handle": to_handle,
        "amount": payment_row["amount"],
        "currency": currency,
        "note": payment_row["note"],
        "visibility": payment_row["visibility"],
        "request_id": payment_row["request_id"],
        "authorization_id": authorization_id,
        "created_at": payment_row["created_at"],
    }


def create_authorization(
    user_id: str,
    idempotency_key: str,
    to_handle: str,
    amount: object,
    note: object = "",
    visibility: object = "public",
) -> tuple[dict, int]:
    """Open a stage-2 authorisation hold (spec §API ``POST /authorizations``).

    Behaviour:
      * First use of ``(user_id, endpoint, idempotency_key)`` → validates
        the payload, opens the hold against the caller's *available*
        funds, persists an ``idempotency_keys`` row alongside, returns
        ``(response_dict, 201)``.
      * Same key + same canonical body → replays the stored ``201``
        response verbatim (``(response_dict, 200)``).
      * Same key + different canonical body → ``409
        idempotency_key_reuse``.
    """
    conn = db_mod.get_connection()
    body = {
        "to_handle": to_handle,
        "amount": amount,
        "note": note,
        "visibility": visibility,
    }

    decision = decide(conn, user_id, CREATE_ENDPOINT, idempotency_key, body)
    if decision.is_replay:
        assert decision.response_payload is not None
        return decision.response_payload, 200
    if decision.is_conflict:
        raise_error(
            409,
            "idempotency_key_reuse",
            "same Idempotency-Key with a different request body",
        )

    amount_i = validate_amount(amount)

    if not isinstance(note, str):
        raise_error(422, "validation_failed", "note must be a string")
    note_str = note
    if len(note_str) > MAX_NOTE_LEN:
        raise_error(
            422, "validation_failed", "note must be at most 200 characters"
        )

    if not isinstance(visibility, str) or visibility not in ALLOWED_VISIBILITY:
        raise_error(
            422,
            "validation_failed",
            "visibility must be 'public' or 'private'",
        )

    caller_row = conn.execute(
        "SELECT id, handle FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    if caller_row is None:
        raise_error(401, "unauthenticated", "user not found")

    if to_handle == caller_row["handle"]:
        raise_error(
            422, "self_payment", "cannot authorize a payment to yourself"
        )

    to_row = conn.execute(
        "SELECT id, handle FROM users WHERE handle = ?", (to_handle,)
    ).fetchone()
    if to_row is None:
        raise_error(404, "not_found", "recipient user not found")

    currency = _currency(conn)
    ttl_seconds = _ttl_seconds(conn)
    now = _utc_now()
    now_iso = now.isoformat(timespec="seconds")
    expires_at = _compute_expires_at(now, ttl_seconds)

    total, available, _held = authz_repo.available_for_user(
        conn, user_id, now_iso=now_iso
    )
    if available < amount_i:
        raise_error(
            409,
            "insufficient_funds",
            "not enough available balance",
        )

    body_hash = canonical_body_hash(body)
    authorization_id = _new_authorization_id()

    with write_transaction(conn) as cur:
        cur.execute("SELECT 1")
        authz_repo.insert_authorization(
            conn,
            authorization_id=authorization_id,
            from_user_id=user_id,
            to_user_id=to_row["id"],
            amount=amount_i,
            note=note_str,
            visibility=visibility,
            expires_at=expires_at,
            created_at=now_iso,
        )
        cur.execute(
            "INSERT INTO idempotency_keys("
            "  user_id, endpoint, key, request_body_hash,"
            "  response_status, response_body, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                CREATE_ENDPOINT,
                idempotency_key,
                body_hash,
                201,
                "",
                now_iso,
            ),
        )

    record = authz_repo.get_authorization(
        conn, authorization_id, now_iso=now_iso
    )
    response = _build_create_response(
        record,
        from_handle=caller_row["handle"],
        to_handle=to_row["handle"],
        currency=currency,
        expires_at=expires_at,
        captures=[],
    )

    idempotency_update_response(
        conn,
        user_id=user_id,
        endpoint=CREATE_ENDPOINT,
        key=idempotency_key,
        response_status=201,
        response_body=json.dumps(response),
    )

    return response, 201


def _coerce_capture_amount(amount: object) -> int | None:
    if amount is None:
        return None
    try:
        return validate_amount(amount)
    except AmountError as exc:
        raise_error(422, "validation_failed", str(exc))


def _coerce_final_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    raise_error(
        422,
        "validation_failed",
        "final must be a boolean when provided",
    )


def capture_authorization(
    user_id: str,
    idempotency_key: str,
    authorization_id: str,
    amount: object = None,
    final: object = True,
) -> tuple[dict, int]:
    """Capture against an open authorisation (spec §API ``POST /authorizations/{id}/capture``).

    Only the receiver (the ``to_user_id``) may capture. The captured
    amount must not exceed the remaining (uncaptured) amount. The
    response is the created **payment** (in the same shape ``POST
    /payments`` returns) with ``authorization_id`` set.

    Idempotency:
      * First use of ``(user_id, endpoint, idempotency_key)`` →
        performs the capture and returns ``(response, 201)``.
      * Same key + same canonical body → replays the stored ``201``.
      * Same key + different canonical body → ``409
        idempotency_key_reuse``.
    """
    conn = db_mod.get_connection()
    body = {"amount": amount, "final": final}

    decision = decide(conn, user_id, CAPTURE_ENDPOINT, idempotency_key, body)
    if decision.is_replay:
        assert decision.response_payload is not None
        return decision.response_payload, 200
    if decision.is_conflict:
        raise_error(
            409,
            "idempotency_key_reuse",
            "same Idempotency-Key with a different request body",
        )

    amount_i: int | None = _coerce_capture_amount(amount)
    final_b = _coerce_final_flag(final)

    now_iso = _utc_now_iso()
    try:
        record = authz_repo.get_authorization(
            conn, authorization_id, now_iso=now_iso
        )
    except AuthorizationNotFound:
        raise_error(404, "not_found", "authorization not found")

    if record.to_user_id != user_id:
        raise_error(
            403,
            "forbidden",
            "only the receiver may capture this authorization",
        )

    if record.status != "open":
        raise_error(
            409,
            "authorization_not_open",
            "authorization is not open",
        )

    if record.expires_at <= now_iso:
        raise_error(
            409,
            "authorization_expired",
            "authorization has expired",
        )

    remaining = max(0, record.amount - record.captured_amount)
    if amount_i is None:
        amount_i = remaining
    if amount_i < 1:
        raise_error(422, "validation_failed", "amount must be at least 1")
    if amount_i > remaining:
        raise_error(
            422,
            "capture_exceeds_authorization",
            f"amount {amount_i} exceeds remaining {remaining}",
        )

    currency = _currency(conn)
    payer = conn.execute(
        "SELECT id, handle, balance FROM users WHERE id = ?",
        (record.from_user_id,),
    ).fetchone()
    if payer is None:
        raise_error(404, "not_found", "payer user not found")

    receiver = conn.execute(
        "SELECT id, handle FROM users WHERE id = ?",
        (record.to_user_id,),
    ).fetchone()
    if receiver is None:
        raise_error(404, "not_found", "receiver user not found")

    if payer["balance"] < amount_i:
        raise_error(
            409,
            "insufficient_funds",
            "payer balance is below the captured amount",
        )

    payment_id = _new_payment_id()
    body_hash = canonical_body_hash(body)

    with write_transaction(conn) as cur:
        cur.execute(
            "UPDATE users SET balance = balance - ? WHERE id = ?",
            (amount_i, payer["id"]),
        )
        cur.execute(
            "UPDATE users SET balance = balance + ? WHERE id = ?",
            (amount_i, receiver["id"]),
        )
        cur.execute(
            "INSERT INTO payments("
            "  id, from_user_id, to_user_id, amount, note,"
            "  visibility, request_id, settlement_id, authorization_id, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                payment_id,
                payer["id"],
                receiver["id"],
                amount_i,
                record.note,
                record.visibility,
                None,
                None,
                authorization_id,
                now_iso,
            ),
        )
        authz_repo.add_to_captured_amount(
            conn, authorization_id, amount_i
        )
        captures = authz_repo.list_captures(conn, authorization_id)
        next_seq = len(captures) + 1
        authz_repo.insert_capture(
            conn,
            authorization_id=authorization_id,
            seq=next_seq,
            payment_id=payment_id,
            amount=amount_i,
            created_at=now_iso,
        )
        authz_repo.set_latest_payment(conn, authorization_id, payment_id)
        new_captured = record.captured_amount + amount_i
        closed = final_b and new_captured >= record.amount
        if closed:
            authz_repo.set_status(conn, authorization_id, "captured")
        cur.execute(
            "INSERT INTO idempotency_keys("
            "  user_id, endpoint, key, request_body_hash,"
            "  response_status, response_body, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                CAPTURE_ENDPOINT,
                idempotency_key,
                body_hash,
                201,
                "",
                now_iso,
            ),
        )

    payment_row = conn.execute(
        "SELECT * FROM payments WHERE id = ?", (payment_id,)
    ).fetchone()
    response = _build_payment_response(
        payment_row,
        from_handle=payer["handle"],
        to_handle=receiver["handle"],
        currency=currency,
        authorization_id=authorization_id,
    )

    idempotency_update_response(
        conn,
        user_id=user_id,
        endpoint=CAPTURE_ENDPOINT,
        key=idempotency_key,
        response_status=201,
        response_body=json.dumps(response),
    )

    return response, 201


__all__ = [
    "CREATE_ENDPOINT",
    "CAPTURE_ENDPOINT",
    "create_authorization",
    "capture_authorization",
    "MAX_NOTE_LEN",
    "ALLOWED_VISIBILITY",
]
