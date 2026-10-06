"""Repository for stage-2 authorisations.

Persistence and read operations against the ``authorizations`` and
``authorization_captures`` tables. The repository never mutates money
balances directly: capturing a hold is a higher-level operation that
must also debit/credit wallets; this module only owns the row-shaped
state. Stage-2 service code will compose those operations against the
existing stage-1 ``BEGIN IMMEDIATE`` write-transaction wrapper from
:mod:`app.sqlite_utils.transaction`.

All read methods honour the spec's expiry-by-clock rule: an
authorisation whose ``expires_at`` is at or before ``now`` is reported
with ``status="expired"`` even if its row still says ``"open"`` in the
database. No background sweeper is required — the spec only requires
that reads and writes *reflect* expiry, not that the row be updated.
"""
from __future__ import annotations

import sqlite3
from typing import Iterable

from .model import AuthorizationCaptureRecord, AuthorizationRecord

__all__ = [
    "AuthorizationNotFound",
    "AuthorizationSeedError",
    "insert_authorization",
    "get_authorization",
    "list_authorizations_for_user",
    "held_for_user",
    "available_for_user",
    "insert_capture",
    "list_captures",
    "validate_seed_authorizations",
    "seed_authorization",
    "is_visible_expired",
    "presentation_fields",
    "set_status",
    "set_latest_payment",
    "add_to_captured_amount",
]


class AuthorizationNotFound(LookupError):
    """Raised when no authorisation matches the requested identifier."""


class AuthorizationSeedError(ValueError):
    """Raised when seeded authorisations fail spec validation (stage-2)."""


def _row_to_record(row: sqlite3.Row, now_iso: str) -> AuthorizationRecord:
    status = row["status"]
    if status == "open" and row["expires_at"] <= now_iso:
        status = "expired"
    return AuthorizationRecord(
        id=row["id"],
        from_user_id=row["from_user_id"],
        to_user_id=row["to_user_id"],
        amount=row["amount"],
        captured_amount=row["captured_amount"],
        note=row["note"],
        visibility=row["visibility"],
        status=status,
        expires_at=row["expires_at"],
        latest_payment_id=row["latest_payment_id"],
        created_at=row["created_at"],
    )


def insert_authorization(
    conn: sqlite3.Connection,
    *,
    authorization_id: str,
    from_user_id: str,
    to_user_id: str,
    amount: int,
    note: str = "",
    visibility: str = "public",
    expires_at: str,
    created_at: str,
) -> AuthorizationRecord:
    """Persist a new authorisation in ``open`` status.

    The capture side (``captured_amount`` / ``latest_payment_id``) starts
    at zero. The caller is responsible for wrapping the call in
    ``BEGIN IMMEDIATE`` if needed.
    """
    if amount < 1:
        raise ValueError("authorization.amount must be at least 1")
    conn.execute(
        "INSERT INTO authorizations("
        "  id, from_user_id, to_user_id, amount, captured_amount,"
        "  note, visibility, status, expires_at, latest_payment_id, created_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            authorization_id,
            from_user_id,
            to_user_id,
            amount,
            0,
            note,
            visibility,
            "open",
            expires_at,
            None,
            created_at,
        ),
    )
    return AuthorizationRecord(
        id=authorization_id,
        from_user_id=from_user_id,
        to_user_id=to_user_id,
        amount=amount,
        captured_amount=0,
        note=note,
        visibility=visibility,
        status="open",
        expires_at=expires_at,
        latest_payment_id=None,
        created_at=created_at,
    )


def get_authorization(
    conn: sqlite3.Connection,
    authorization_id: str,
    *,
    now_iso: str,
) -> AuthorizationRecord:
    """Load an authorisation by id, applying expiry-by-clock.

    Raises ``AuthorizationNotFound`` when no row matches.
    """
    row = conn.execute(
        "SELECT id, from_user_id, to_user_id, amount, captured_amount,"
        "  note, visibility, status, expires_at, latest_payment_id, created_at "
        "FROM authorizations WHERE id = ?",
        (authorization_id,),
    ).fetchone()
    if row is None:
        raise AuthorizationNotFound(authorization_id)
    return _row_to_record(row, now_iso)


def list_authorizations_for_user(
    conn: sqlite3.Connection,
    user_id: str,
    *,
    direction: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    now_iso: str,
) -> tuple[list[AuthorizationRecord], bool]:
    """Return authorisations involving ``user_id``, newest first.

    Filters:
      * ``direction`` is ``"outgoing"`` (caller is the payer / ``from``),
        ``"incoming"`` (caller is the receiver / ``to``) or ``None``
        for both. An unknown value is rejected.
      * ``status`` is one of ``open | captured | voided | expired`` or
        ``None`` for all. The expiry-by-clock transformation is applied
        so an authorisation whose row says ``open`` but whose
        ``expires_at`` is past matches ``status="expired"`` (and never
        matches ``status="open"`` when filtered). Unknown statuses are
        rejected.

    Returns ``(rows, has_more)`` where ``has_more`` is True when at
    least one additional row exists past ``offset + limit``.
    """
    if not isinstance(limit, int) or limit < 1 or limit > 200:
        raise ValueError("limit must be an integer from 1 to 200")
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be 0 or more")
    if direction is not None and direction not in {"incoming", "outgoing"}:
        raise ValueError("direction must be 'incoming' or 'outgoing'")
    if status is not None and status not in {"open", "captured", "voided", "expired"}:
        raise ValueError("status must be one of open|captured|voided|expired")

    conds: list[str] = []
    params: list[object] = []
    if direction == "outgoing":
        conds.append("from_user_id = ?")
        params.append(user_id)
    elif direction == "incoming":
        conds.append("to_user_id = ?")
        params.append(user_id)
    else:
        conds.append("(from_user_id = ? OR to_user_id = ?)")
        params.extend([user_id, user_id])
    where_clause = " AND ".join(conds)

    # Two-step query: fetch the candidate window, apply the in-Python
    # expiry-by-clock projection, then re-filter by the (possibly
    # transformed) status. The expiry projection is intentionally in
    # Python so an "open" row whose expires_at is past gets reclassified
    # to "expired" and matches `status="expired"` filters.
    fetch = limit + offset + 1
    rows = conn.execute(
        f"SELECT id, from_user_id, to_user_id, amount, captured_amount,"
        f"  note, visibility, status, expires_at, latest_payment_id, created_at "
        f"FROM authorizations WHERE {where_clause} "
        f"ORDER BY created_at DESC, id DESC LIMIT ?",
        (*params, fetch),
    ).fetchall()

    records = [_row_to_record(row, now_iso) for row in rows]
    if status is not None:
        records = [r for r in records if r.status == status]

    page = records[offset : offset + limit]
    has_more = len(records) > offset + limit
    return page, has_more


def held_for_user(
    conn: sqlite3.Connection,
    user_id: str,
    *,
    now_iso: str,
) -> int:
    """Sum the ``amount`` of every still-open, non-expired hold for ``user_id``.

    Only ``status="open"`` rows whose ``expires_at`` is strictly greater
    than ``now`` count. Captured / voided / expired holds do not.
    """
    row = conn.execute(
        "SELECT COALESCE(SUM(amount), 0) AS held FROM authorizations "
        "WHERE from_user_id = ? AND status = 'open' AND expires_at > ?",
        (user_id, now_iso),
    ).fetchone()
    return int(row["held"])


def available_for_user(
    conn: sqlite3.Connection,
    user_id: str,
    *,
    now_iso: str,
) -> tuple[int, int, int]:
    """Return ``(total, available, held)`` for ``user_id``.

    ``total`` is the user's stored balance (which by spec invariant
    equals ``balance``). ``held`` is the sum of non-expired open holds
    where ``user_id`` is the payer. ``available = total - held`` and is
    clamped to zero on the low side, matching the spec rule that
    ``available`` is never negative.
    """
    user_row = conn.execute(
        "SELECT balance FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    if user_row is None:
        raise AuthorizationNotFound(user_id)
    total = int(user_row["balance"])
    held = held_for_user(conn, user_id, now_iso=now_iso)
    available = total - held
    if available < 0:
        available = 0
    return total, available, held


def insert_capture(
    conn: sqlite3.Connection,
    *,
    authorization_id: str,
    seq: int,
    payment_id: str,
    amount: int,
    created_at: str,
) -> AuthorizationCaptureRecord:
    """Persist a new capture row for ``authorization_id``.

    Caller is responsible for also updating
    ``authorizations.captured_amount`` and ``latest_payment_id`` and
    for the underlying wallet movements; this function only records
    the per-capture history row that supports extended capture mode.
    """
    if seq < 1:
        raise ValueError("capture.seq must be >= 1")
    if amount < 1:
        raise ValueError("capture.amount must be at least 1")
    conn.execute(
        "INSERT INTO authorization_captures("
        "  authorization_id, seq, payment_id, amount, created_at"
        ") VALUES (?, ?, ?, ?, ?)",
        (authorization_id, seq, payment_id, amount, created_at),
    )
    return AuthorizationCaptureRecord(
        authorization_id=authorization_id,
        seq=seq,
        payment_id=payment_id,
        amount=amount,
        created_at=created_at,
    )


def list_captures(
    conn: sqlite3.Connection, authorization_id: str
) -> list[AuthorizationCaptureRecord]:
    """Return all capture rows for ``authorization_id`` in seq order."""
    rows = conn.execute(
        "SELECT authorization_id, seq, payment_id, amount, created_at "
        "FROM authorization_captures WHERE authorization_id = ? "
        "ORDER BY seq ASC",
        (authorization_id,),
    ).fetchall()
    return [
        AuthorizationCaptureRecord(
            authorization_id=row["authorization_id"],
            seq=row["seq"],
            payment_id=row["payment_id"],
            amount=row["amount"],
            created_at=row["created_at"],
        )
        for row in rows
    ]


def is_visible_expired(record: AuthorizationRecord, now_iso: str) -> bool:
    """True when an open authorisation has crossed its expiry deadline."""
    return record.status == "expired" or (
        record.status == "open" and record.expires_at <= now_iso
    )


def validate_seed_authorizations(
    conn: sqlite3.Connection,
    raw_authorizations: Iterable[dict],
    *,
    now_iso: str,
) -> None:
    """Validate the ``authorizations`` array of a reset fixture.

    Spec §Model requires:
      * each row's ``status`` is one of ``open | captured | voided | expired``;
      * a sum of seeded unexpired open holds per payer that exceeds the
        payer's seeded balance is a reset error.

    Raises ``AuthorizationSeedError`` with a human-readable message on
    the first violation. Returns ``None`` when every row is valid.
    """
    held_by_user: dict[str, int] = {}
    balances: dict[str, int] = {}

    for row in raw_authorizations:
        if not isinstance(row, dict):
            raise AuthorizationSeedError(
                "seeded authorization must be an object"
            )
        try:
            auth_id = str(row["id"])
            from_user_id = str(row["from_user_id"])
            to_user_id = str(row["to_user_id"])
            amount = int(row["amount"])
            status = str(row.get("status", "open"))
            expires_at = str(row["expires_at"])
        except KeyError as exc:
            raise AuthorizationSeedError(
                f"seeded authorization missing field {exc.args[0]!r}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise AuthorizationSeedError(
                f"seeded authorization has invalid field type: {exc}"
            ) from exc

        if status not in {"open", "captured", "voided", "expired"}:
            raise AuthorizationSeedError(
                f"seeded authorization {auth_id} has invalid status "
                f"{status!r}"
            )
        if amount < 1:
            raise AuthorizationSeedError(
                f"seeded authorization {auth_id} amount must be at least 1"
            )
        if from_user_id == to_user_id:
            raise AuthorizationSeedError(
                f"seeded authorization {auth_id} cannot hold against itself"
            )

        # Sum only open-and-not-expired holds per payer.
        if status == "open" and expires_at > now_iso:
            held_by_user[from_user_id] = held_by_user.get(from_user_id, 0) + amount

    # Resolve each payer's balance from the users table (which has already
    # been seeded by the caller).
    for user_id in held_by_user:
        user_row = conn.execute(
            "SELECT balance FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if user_row is None:
            raise AuthorizationSeedError(
                f"seeded authorization references unknown user {user_id!r}"
            )
        balances[user_id] = int(user_row["balance"])

    for user_id, held in held_by_user.items():
        if held > balances.get(user_id, 0):
            raise AuthorizationSeedError(
                f"seeded open holds for user {user_id!r} sum to {held} "
                f"which exceeds the user's seeded balance "
                f"{balances[user_id]}"
            )


def seed_authorization(
    conn: sqlite3.Connection,
    *,
    authorization_id: str,
    from_user_id: str,
    to_user_id: str,
    amount: int,
    note: str,
    visibility: str,
    status: str,
    expires_at: str,
    created_at: str,
    latest_payment_id: str | None = None,
    captured_amount: int | None = None,
) -> AuthorizationRecord:
    """Insert a seeded authorisation as supplied by a reset fixture.

    Unlike :func:`insert_authorization`, this function honours the
    caller's ``status`` and ``captured_amount`` (defaulting to ``0``)
    so a fixture can replay a partially-captured state without
    rewriting the underlying wallet balances. ``captured_amount`` is
    clamped to ``[0, amount]``.
    """
    if status not in {"open", "captured", "voided", "expired"}:
        raise AuthorizationSeedError(
            f"invalid seed status {status!r}"
        )
    if amount < 1:
        raise AuthorizationSeedError("seed amount must be at least 1")
    if captured_amount is None:
        captured_amount = 0
    if captured_amount < 0:
        captured_amount = 0
    if captured_amount > amount:
        captured_amount = amount
    conn.execute(
        "INSERT INTO authorizations("
        "  id, from_user_id, to_user_id, amount, captured_amount,"
        "  note, visibility, status, expires_at, latest_payment_id, created_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            authorization_id,
            from_user_id,
            to_user_id,
            amount,
            captured_amount,
            note,
            visibility,
            status,
            expires_at,
            latest_payment_id,
            created_at,
        ),
    )
    return AuthorizationRecord(
        id=authorization_id,
        from_user_id=from_user_id,
        to_user_id=to_user_id,
        amount=amount,
        captured_amount=captured_amount,
        note=note,
        visibility=visibility,
        status=status,
        expires_at=expires_at,
        latest_payment_id=latest_payment_id,
        created_at=created_at,
    )


def presentation_fields(
    record: AuthorizationRecord,
    captures: list[AuthorizationCaptureRecord],
    *,
    now_iso: str,
) -> dict:
    """Compute the spec-shaped response fields for an authorisation.

    Spec §API requires every authorisation response to include the
    cumulative ``captured_amount``, the ``payment_id`` of the latest
    capture (or ``None`` if none have happened), the ordered list
    ``payment_ids``, the still-held ``remaining_amount`` (zero when
    the authorisation is closed), and the *visible* ``status`` —
    applying the expiry-by-clock rule. The record's stored ``status``
    is not mutated; only this view reflects expiry.
    """
    visible_status = record.status
    if visible_status == "open" and record.expires_at <= now_iso:
        visible_status = "expired"
    payment_ids = [c.payment_id for c in sorted(captures, key=lambda c: c.seq)]
    latest_payment_id = record.latest_payment_id
    if captures:
        latest = max(captures, key=lambda c: c.seq)
        latest_payment_id = latest.payment_id
    remaining_amount = max(0, record.amount - record.captured_amount)
    if visible_status in {"captured", "voided", "expired"}:
        remaining_amount = 0
    return {
        "captured_amount": record.captured_amount,
        "payment_id": latest_payment_id,
        "payment_ids": payment_ids,
        "remaining_amount": remaining_amount,
        "status": visible_status,
    }


def set_status(
    conn: sqlite3.Connection,
    authorization_id: str,
    status: str,
) -> None:
    """Update an authorisation's ``status`` column.

    The persistence layer does not enforce lifecycle ordering; that is
    the service layer's job. This function only persists the
    transition. Pass ``"expired"`` when a write-time expiry check
    confirms the deadline has passed; reads continue to apply the
    expiry-by-clock rule on top of the stored value.
    """
    if status not in {"open", "captured", "voided", "expired"}:
        raise ValueError(
            f"authorization.status must be open|captured|voided|expired, "
            f"got {status!r}"
        )
    conn.execute(
        "UPDATE authorizations SET status = ? WHERE id = ?",
        (status, authorization_id),
    )


def set_latest_payment(
    conn: sqlite3.Connection,
    authorization_id: str,
    payment_id: str,
) -> None:
    """Record the most-recent capture's payment id on the authorisation row."""
    conn.execute(
        "UPDATE authorizations SET latest_payment_id = ? WHERE id = ?",
        (payment_id, authorization_id),
    )


def add_to_captured_amount(
    conn: sqlite3.Connection,
    authorization_id: str,
    delta: int,
) -> int:
    """Increment ``captured_amount`` by ``delta`` and return the new value.

    The CHECK constraint on the table guards against the cumulative
    going negative or exceeding ``amount``; the caller is responsible
    for staying within those bounds (the service layer is the place
    where the bounds are enforced). This function does not modify
    ``status`` — the lifecycle owner decides when to flip the
    authorisation to ``captured`` / ``voided`` / ``expired``.
    """
    if delta < 1:
        raise ValueError("captured_amount delta must be >= 1")
    conn.execute(
        "UPDATE authorizations SET captured_amount = captured_amount + ? "
        "WHERE id = ?",
        (delta, authorization_id),
    )
    row = conn.execute(
        "SELECT captured_amount FROM authorizations WHERE id = ?",
        (authorization_id,),
    ).fetchone()
    return int(row["captured_amount"])
