"""Payment request creation and lifecycle with idempotency (spec §8).

* ``POST /requests``               — creates a pending request (idempotent).
* ``POST /requests/{id}/pay``     — idempotent write; the payer settles a
  pending request. Atomically debits/credits balances, inserts a payment
  (activity log) and flips the request to ``paid`` inside one
  ``BEGIN IMMEDIATE`` transaction.
* ``POST /requests/{id}/decline`` — payer-only transition; the request
  becomes ``declined``. Declining twice is a 200 no-op (no body key).
* ``POST /requests/{id}/cancel``  — requester-only transition; the request
  becomes ``cancelled``. Cancelling twice is a 200 no-op (no body key).

The lifecycle transitions share the same invariants as the rest of the
service: write paths go through ``BEGIN IMMEDIATE``; the canonical body
hash for ``pay`` is computed via
:func:`app.idempotency.canonical_hash.canonical_body_hash` and the
decision is resolved via :func:`app.idempotency.decision.decide`.
``pay``'s ``insufficient_funds`` check honours the stage-2 ``available``
balance (spec stage-2 §API) via
:func:`app.authorizations.repository.available_for_user`.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from .. import db as db_mod
from ..authorizations import repository as authz_repo
from ..errors import raise_error
from ..idempotency.canonical_hash import canonical_body_hash
from ..idempotency.decision import decide as idem_decide, update_response as idem_update_response
from ..models.balance import validate_amount
from ..sqlite_utils.transaction import write_transaction

ENDPOINT = "POST /requests"
PAY_ENDPOINT = "POST /requests/{id}/pay"
DECLINE_ENDPOINT = "POST /requests/{id}/decline"
CANCEL_ENDPOINT = "POST /requests/{id}/cancel"
MAX_NOTE_LEN = 200
ALLOWED_STATUSES = frozenset({"pending", "paid", "declined", "cancelled"})
ALLOWED_DIRECTIONS = frozenset({"incoming", "outgoing"})
ALLOWED_PAY_VISIBILITY = frozenset({"public", "private"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash_body(body: dict) -> str:
    normalized = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _service_currency(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT value FROM service_meta WHERE key = 'currency'"
    ).fetchone()
    return str(row["value"]) if row else "EUR"


def _build_response(request_row: dict, user_requester: dict, user_payer: dict) -> dict:
    return {
        "request_id": request_row["id"],
        "requester_id": request_row["requester_id"],
        "requester_handle": user_requester["handle"],
        "payer_id": request_row["payer_id"],
        "payer_handle": user_payer["handle"],
        "amount": request_row["amount"],
        "currency": user_requester["currency"],
        "note": request_row["note"],
        "status": request_row["status"],
        "payment_id": request_row["payment_id"],
        "created_at": request_row["created_at"],
    }


def _build_payment_response_for_request(payment_row: dict, user_from: dict, user_to: dict) -> dict:
    return {
        "payment_id": payment_row["id"],
        "from_user_id": payment_row["from_user_id"],
        "from_handle": user_from["handle"],
        "to_user_id": payment_row["to_user_id"],
        "to_handle": user_to["handle"],
        "amount": payment_row["amount"],
        "currency": user_from["currency"],
        "note": payment_row["note"],
        "visibility": payment_row["visibility"],
        "request_id": payment_row["request_id"],
        "created_at": payment_row["created_at"],
    }


def _build_request_status_response(request_row: dict, user_requester: dict, user_payer: dict) -> dict:
    return {
        "request_id": request_row["id"],
        "requester_id": request_row["requester_id"],
        "requester_handle": user_requester["handle"],
        "payer_id": request_row["payer_id"],
        "payer_handle": user_payer["handle"],
        "amount": request_row["amount"],
        "currency": user_requester["currency"],
        "note": request_row["note"],
        "status": request_row["status"],
        "payment_id": request_row["payment_id"],
        "created_at": request_row["created_at"],
    }


def create_request(
    user_id: str,
    idempotency_key: str,
    payer_handle: str,
    amount: object,
    note: object = "",
) -> tuple[dict, int]:
    conn = db_mod.get_connection()
    body = {
        "payer_handle": payer_handle,
        "amount": amount,
        "note": note,
    }
    body_hash = _hash_body(body)

    existing = conn.execute(
        "SELECT response_status, response_body FROM idempotency_keys "
        "WHERE user_id = ? AND endpoint = ? AND key = ?",
        (user_id, ENDPOINT, idempotency_key),
    ).fetchone()

    if existing is not None:
        stored_hash = conn.execute(
            "SELECT request_body_hash FROM idempotency_keys "
            "WHERE user_id = ? AND endpoint = ? AND key = ?",
            (user_id, ENDPOINT, idempotency_key),
        ).fetchone()["request_body_hash"]
        if stored_hash != body_hash:
            raise_error(409, "idempotency_key_reuse", "same key, different request body")
        return json.loads(existing["response_body"]), 200

    amount_i = validate_amount(amount)
    if not isinstance(note, str):
        raise_error(422, "validation_failed", "note must be a string")
    note_str = note if note else ""
    if len(note_str) > MAX_NOTE_LEN:
        raise_error(422, "validation_failed", "note must be at most 200 characters")

    requester = conn.execute(
        "SELECT id, handle, currency FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    if requester is None:
        raise_error(401, "unauthenticated", "requester user not found")

    if payer_handle == requester["handle"]:
        raise_error(422, "self_request", "cannot request payment from yourself")

    payer = conn.execute(
        "SELECT id, handle, currency FROM users WHERE handle = ?", (payer_handle,)
    ).fetchone()
    if payer is None:
        raise_error(404, "not_found", "payer user not found")

    request_id = f"rq_{secrets.token_hex(8)}"
    created_at = _utc_now_iso()

    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO payment_requests(id, requester_id, payer_id, amount, note, status, payment_id, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (request_id, user_id, payer["id"], amount_i, note_str, "pending", None, created_at),
        )
        conn.execute(
            "INSERT INTO idempotency_keys(user_id, endpoint, key, request_body_hash, response_status, response_body, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?)",
            (user_id, ENDPOINT, idempotency_key, body_hash, 201, "", created_at),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    request_row = conn.execute(
        "SELECT * FROM payment_requests WHERE id = ?", (request_id,)
    ).fetchone()
    response = _build_response(request_row, requester, payer)
    response_json = json.dumps(response)

    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "UPDATE idempotency_keys SET response_body = ?, response_status = ? "
        "WHERE user_id = ? AND endpoint = ? AND key = ?",
        (response_json, 201, user_id, ENDPOINT, idempotency_key),
    )
    conn.execute("COMMIT")

    return response, 201


def _build_request_row(row: dict, requester_handle: str, payer_handle: str, currency: str) -> dict:
    return {
        "request_id": row["id"],
        "requester_id": row["requester_id"],
        "requester_handle": requester_handle,
        "payer_id": row["payer_id"],
        "payer_handle": payer_handle,
        "amount": row["amount"],
        "currency": currency,
        "note": row["note"],
        "status": row["status"],
        "payment_id": row["payment_id"],
        "created_at": row["created_at"],
    }


def list_requests(
    user_id: str,
    direction: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    if not isinstance(limit, int) or limit < 1 or limit > 200:
        raise_error(422, "validation_failed", "limit must be an integer from 1 to 200")
    if not isinstance(offset, int) or offset < 0:
        raise_error(422, "validation_failed", "offset must be 0 or more")

    if direction is not None and direction not in ALLOWED_DIRECTIONS:
        raise_error(422, "validation_failed", "direction must be 'incoming' or 'outgoing'")
    if status is not None and status not in ALLOWED_STATUSES:
        raise_error(422, "validation_failed", "status must be one of: pending, paid, declined, cancelled")

    conn = db_mod.get_connection()

    user = conn.execute("SELECT handle FROM users WHERE id = ?", (user_id,)).fetchone()
    if user is None:
        raise_error(401, "unauthenticated", "user not found")

    conditions = []
    params = []

    if direction == "incoming":
        conditions.append("payer_id = ?")
        params.append(user_id)
    elif direction == "outgoing":
        conditions.append("requester_id = ?")
        params.append(user_id)
    else:
        conditions.append("(requester_id = ? OR payer_id = ?)")
        params.extend([user_id, user_id])

    if status is not None:
        conditions.append("status = ?")
        params.append(status)

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    currency_row = conn.execute("SELECT value FROM service_meta WHERE key = 'currency'").fetchone()
    currency = currency_row["value"] if currency_row else "EUR"

    count_sql = f"SELECT COUNT(*) as total FROM payment_requests WHERE {where_clause}"
    total = conn.execute(count_sql, params).fetchone()["total"]

    sql = f"""
        SELECT pr.*, 
               u_req.handle as requester_handle, 
               u_pay.handle as payer_handle
        FROM payment_requests pr
        JOIN users u_req ON pr.requester_id = u_req.id
        JOIN users u_pay ON pr.payer_id = u_pay.id
        WHERE {where_clause}
        ORDER BY pr.created_at DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit + 1, offset])
    rows = conn.execute(sql, params).fetchall()

    has_more = len(rows) > limit
    if has_more:
        rows = rows[:limit]

    requests_list = []
    for row in rows:
        requests_list.append(_build_request_row(dict(row), row["requester_handle"], row["payer_handle"], currency))

    return {
        "requests": requests_list,
        "has_more": has_more,
    }


def pay_request(
    user_id: str,
    idempotency_key: str,
    request_id: str,
    visibility: str = "public",
) -> tuple[dict, int]:
    conn = db_mod.get_connection()
    body = {
        "visibility": visibility,
    }
    body_hash = canonical_body_hash(body)

    existing = conn.execute(
        "SELECT response_status, response_body FROM idempotency_keys "
        "WHERE user_id = ? AND endpoint = ? AND key = ?",
        (user_id, PAY_ENDPOINT, idempotency_key),
    ).fetchone()

    if existing is not None:
        stored_hash = conn.execute(
            "SELECT request_body_hash FROM idempotency_keys "
            "WHERE user_id = ? AND endpoint = ? AND key = ?",
            (user_id, PAY_ENDPOINT, idempotency_key),
        ).fetchone()["request_body_hash"]
        if stored_hash != body_hash:
            raise_error(409, "idempotency_key_reuse", "same key, different request body")
        return json.loads(existing["response_body"]), 200

    if not isinstance(visibility, str) or visibility not in ALLOWED_PAY_VISIBILITY:
        raise_error(422, "validation_failed", "visibility must be 'public' or 'private'")

    request_row = conn.execute(
        "SELECT * FROM payment_requests WHERE id = ?", (request_id,)
    ).fetchone()
    if request_row is None:
        raise_error(404, "not_found", "request not found")

    if request_row["payer_id"] != user_id:
        raise_error(403, "forbidden", "only the payer can pay the request")

    if request_row["status"] != "pending":
        raise_error(409, "request_not_pending", f"request is already {request_row['status']}")

    amount = request_row["amount"]
    available = authz_repo.available_for_user(user_id)
    if available < amount:
        raise_error(409, "insufficient_funds", "payer has insufficient available balance")

    requester = conn.execute(
        "SELECT id, handle, currency FROM users WHERE id = ?", (request_row["requester_id"],)
    ).fetchone()
    payer = conn.execute(
        "SELECT id, handle, currency FROM users WHERE id = ?", (user_id,)
    ).fetchone()

    payment_id = f"p_{secrets.token_hex(8)}"
    created_at = _utc_now_iso()
    note_str = request_row["note"] if request_row["note"] else ""

    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO payments(id, from_user_id, to_user_id, amount, note, visibility, request_id, settlement_id, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (payment_id, user_id, request_row["requester_id"], amount, note_str, visibility, request_id, None, created_at),
        )
        conn.execute(
            "UPDATE users SET balance = balance - ? WHERE id = ?",
            (amount, user_id),
        )
        conn.execute(
            "UPDATE users SET balance = balance + ? WHERE id = ?",
            (amount, request_row["requester_id"]),
        )
        conn.execute(
            "UPDATE payment_requests SET status = ?, payment_id = ? WHERE id = ?",
            ("paid", payment_id, request_id),
        )
        conn.execute(
            "INSERT INTO idempotency_keys(user_id, endpoint, key, request_body_hash, response_status, response_body, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?)",
            (user_id, PAY_ENDPOINT, idempotency_key, body_hash, 201, "", created_at),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    payment_row = conn.execute(
        "SELECT * FROM payments WHERE id = ?", (payment_id,)
    ).fetchone()
    response = _build_payment_response_for_request(payment_row, payer, requester)
    response_json = json.dumps(response)

    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "UPDATE idempotency_keys SET response_body = ?, response_status = ? "
        "WHERE user_id = ? AND endpoint = ? AND key = ?",
        (response_json, 201, user_id, PAY_ENDPOINT, idempotency_key),
    )
    conn.execute("COMMIT")

    return response, 201


def decline_request(user_id: str, request_id: str) -> tuple[dict, int]:
    conn = db_mod.get_connection()

    request_row = conn.execute(
        "SELECT * FROM payment_requests WHERE id = ?", (request_id,)
    ).fetchone()
    if request_row is None:
        raise_error(404, "not_found", "request not found")

    if request_row["payer_id"] != user_id:
        raise_error(403, "forbidden", "only the payer can decline the request")

    if request_row["status"] == "declined":
        return {}, 200

    if request_row["status"] != "pending":
        raise_error(409, "request_not_pending", f"request is already {request_row['status']}")

    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE payment_requests SET status = ? WHERE id = ?",
            ("declined", request_id),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    requester = conn.execute(
        "SELECT id, handle, currency FROM users WHERE id = ?", (request_row["requester_id"],)
    ).fetchone()
    payer = conn.execute(
        "SELECT id, handle, currency FROM users WHERE id = ?", (user_id,)
    ).fetchone()

    request_row = conn.execute(
        "SELECT * FROM payment_requests WHERE id = ?", (request_id,)
    ).fetchone()

    response = _build_request_status_response(request_row, requester, payer)
    return response, 200


def cancel_request(user_id: str, request_id: str) -> tuple[dict, int]:
    conn = db_mod.get_connection()

    request_row = conn.execute(
        "SELECT * FROM payment_requests WHERE id = ?", (request_id,)
    ).fetchone()
    if request_row is None:
        raise_error(404, "not_found", "request not found")

    if request_row["requester_id"] != user_id:
        raise_error(403, "forbidden", "only the requester can cancel the request")

    if request_row["status"] == "cancelled":
        return {}, 200

    if request_row["status"] != "pending":
        raise_error(409, "request_not_pending", f"request is already {request_row['status']}")

    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE payment_requests SET status = ? WHERE id = ?",
            ("cancelled", request_id),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    requester = conn.execute(
        "SELECT id, handle, currency FROM users WHERE id = ?", (request_row["requester_id"],)
    ).fetchone()
    payer = conn.execute(
        "SELECT id, handle, currency FROM users WHERE id = ?", (request_row["payer_id"],)
    ).fetchone()

    request_row = conn.execute(
        "SELECT * FROM payment_requests WHERE id = ?", (request_id,)
    ).fetchone()

    response = _build_request_status_response(request_row, requester, payer)
    return response, 200


__all__ = ["create_request", "list_requests", "pay_request", "decline_request", "cancel_request"]
