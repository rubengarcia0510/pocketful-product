"""Stage-2 authorisations HTTP API.

Endpoints (spec §API):
  * ``POST /authorizations``                 — open a hold
  * ``POST /authorizations/{id}/captures``   — capture against an open hold

Both endpoints require the ``Idempotency-Key`` header (spec §7) and
return JSON. Both delegate to :mod:`app.services.authorization_service`,
which is the single owner of the idempotency wiring (PDF-36).
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from fastapi import APIRouter, Body, Depends, Header
from fastapi.responses import JSONResponse

from ..auth import current_user_id
from ..errors import raise_error
from ..models.balance import AmountError
from ..services import authorization_service

router = APIRouter()


def _require_idempotency_key(key: Optional[str]) -> str:
    if not key:
        raise_error(
            400,
            "missing_idempotency_key",
            "Idempotency-Key header is required",
        )
    if not isinstance(key, str) or len(key) > 255:
        raise_error(
            422,
            "validation_failed",
            "Idempotency-Key must be 1-255 characters",
        )
    return key


@router.post("/authorizations", status_code=201)
def create_authorization(
    user_id: Annotated[str, Depends(current_user_id)],
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    payload: dict[str, Any] = Body(default_factory=dict),
) -> JSONResponse:
    idempotency_key_s = _require_idempotency_key(idempotency_key)
    try:
        response, status = authorization_service.create_authorization(
            user_id=user_id,
            idempotency_key=idempotency_key_s,
            to_handle=payload.get("to_handle"),
            amount=payload.get("amount"),
            note=payload.get("note", ""),
            visibility=payload.get("visibility", "public"),
        )
    except AmountError as exc:
        raise_error(422, "validation_failed", str(exc))
    return JSONResponse(content=response, status_code=status)


@router.post("/authorizations/{authorization_id}/captures", status_code=201)
def capture_authorization(
    authorization_id: str,
    user_id: Annotated[str, Depends(current_user_id)],
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    payload: dict[str, Any] = Body(default_factory=dict),
) -> JSONResponse:
    idempotency_key_s = _require_idempotency_key(idempotency_key)
    try:
        response, status = authorization_service.capture_authorization(
            user_id=user_id,
            idempotency_key=idempotency_key_s,
            authorization_id=authorization_id,
            amount=payload.get("amount"),
            final=payload.get("final", True),
        )
    except AmountError as exc:
        raise_error(422, "validation_failed", str(exc))
    return JSONResponse(content=response, status_code=status)


__all__ = ["router"]
