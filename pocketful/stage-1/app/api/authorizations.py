"""Stage-2 authorisations HTTP API.

Endpoints (spec §API):
  * ``POST /authorizations``                — open a hold
  * ``POST /authorizations/{id}/captures``  — capture against an open hold
  * ``POST /authorizations/{id}/capture``   — spec-aligned capture alias
  * ``POST /authorizations/{id}/void``      — void an open hold
  * ``GET  /authorizations``                — list caller's authorisations

``POST /authorizations``, ``POST /authorizations/{id}/captures`` and
``POST /authorizations/{id}/capture`` require the ``Idempotency-Key``
header (spec §7). ``POST /authorizations/{id}/void`` does not. All
endpoints return JSON and delegate to
:mod:`app.services.authorization_service`, which is the single owner of
the idempotency wiring (PDF-36) and the lifecycle state machine
(PDF-38).
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from fastapi import APIRouter, Body, Depends, Header, Query
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


def _capture(
    authorization_id: str,
    user_id: str,
    idempotency_key: Optional[str],
    payload: dict[str, Any],
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


@router.post("/authorizations/{authorization_id}/captures", status_code=201)
def capture_authorization(
    authorization_id: str,
    user_id: Annotated[str, Depends(current_user_id)],
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    payload: dict[str, Any] = Body(default_factory=dict),
) -> JSONResponse:
    return _capture(authorization_id, user_id, idempotency_key, payload)


@router.post("/authorizations/{authorization_id}/capture", status_code=201)
def capture_authorization_singular(
    authorization_id: str,
    user_id: Annotated[str, Depends(current_user_id)],
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    payload: dict[str, Any] = Body(default_factory=dict),
) -> JSONResponse:
    """Spec-aligned singular-path alias for ``/captures``.

    The spec uses ``POST /authorizations/{id}/capture`` (singular). The
    plural route ``/captures`` is kept as a deprecated alias for
    backward compatibility with earlier-stage-2 callers and tests.
    """
    return _capture(authorization_id, user_id, idempotency_key, payload)


@router.post("/authorizations/{authorization_id}/void")
def void_authorization(
    authorization_id: str,
    user_id: Annotated[str, Depends(current_user_id)],
) -> JSONResponse:
    response, status = authorization_service.void_authorization(
        user_id=user_id,
        authorization_id=authorization_id,
    )
    return JSONResponse(content=response, status_code=status)


@router.get("/authorizations")
def list_authorizations(
    user_id: Annotated[str, Depends(current_user_id)],
    direction: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> JSONResponse:
    response = authorization_service.list_authorizations(
        user_id=user_id,
        direction=direction,
        status=status,
        limit=limit,
        offset=offset,
    )
    return JSONResponse(content=response)


__all__ = ["router"]
