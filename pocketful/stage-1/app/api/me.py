"""GET /me — returns the authenticated user's public profile, balance and holds.

Stage-2 (spec §API ``GET /me``): the response carries ``balance``,
``total``, ``available`` and ``held``. ``balance`` and ``total`` are
always equal; ``held`` is the sum of the caller's non-expired open
holds; ``available`` is ``total - held`` and is clamped to zero.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends

from .. import db as db_mod
from ..authorizations import repository as authz_repo
from ..auth import current_user
from ..models.user import UserRecord

router = APIRouter()


@router.get("/me")
def me(user: Annotated[UserRecord, Depends(current_user)]) -> dict:
    conn = db_mod.get_connection()
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    total, available, held = authz_repo.available_for_user(
        conn, user.id, now_iso=now_iso
    )
    return {
        "user_id": user.id,
        "display_name": user.display_name,
        "handle": user.handle,
        "balance": user.balance,
        "total": total,
        "available": available,
        "held": held,
        "currency": user.currency,
        "minor_units": user.minor_units,
    }


__all__ = ["router"]
