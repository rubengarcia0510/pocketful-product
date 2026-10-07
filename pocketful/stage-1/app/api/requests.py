"""POST /requests, GET /requests (spec §8)."""

from __future__ import annotations

from typing import Annotated, Any, Optional

from fastapi import APIRouter, Body, Depends, Header, Query
from fastapi.responses import JSONResponse

from ..auth import current_user_id
from ..errors import raise_error
from ..models.balance import AmountError
from ..services import request_service

router = APIRouter()


@router.post("/requests")
def create_request(
    user_id: Annotated[str, Depends(current_user_id)],
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    payload: dict[str, Any] = Body(default_factory=dict),
) -> JSONResponse:
    if not idempotency_key:
        raise_error(400, "missing_idempotency_key", "Idempotency-Key header is required")
    if not isinstance(idempotency_key, str) or len(idempotency_key) > 255:
        raise_error(422, "validation_failed", "Idempotency-Key must be 1-255 characters")

    try:
        response, status = request_service.create_request(
            user_id=user_id,
            idempotency_key=idempotency_key,
            payer_handle=payload.get("payer_handle"),
            amount=payload.get("amount"),
            note=payload.get("note", ""),
        )
    except AmountError as e:
        raise_error(422, "validation_failed", str(e))
    return JSONResponse(content=response, status_code=status)


@router.get("/requests")
def list_requests(
    user_id: Annotated[str, Depends(current_user_id)],
    direction: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> JSONResponse:
    response = request_service.list_requests(
        user_id=user_id,
        direction=direction,
        status=status,
        limit=limit,
        offset=offset,
    )
    return JSONResponse(content=response)


@router.post("/requests/{request_id}/pay")
def pay_request(
    request_id: str,
    user_id: Annotated[str, Depends(current_user_id)],
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    payload: dict[str, Any] = Body(default_factory=dict),
) -> JSONResponse:
    if not idempotency_key:
        raise_error(400, "missing_idempotency_key", "Idempotency-Key header is required")
    if not isinstance(idempotency_key, str) or len(idempotency_key) > 255:
        raise_error(422, "validation_failed", "Idempotency-Key must be 1-255 characters")

    visibility = payload.get("visibility", "public")
    response, status = request_service.pay_request(
        user_id=user_id,
        idempotency_key=idempotency_key,
        request_id=request_id,
        visibility=visibility,
    )
    return JSONResponse(content=response, status_code=status)


@router.post("/requests/{request_id}/decline")
def decline_request(
    request_id: str,
    user_id: Annotated[str, Depends(current_user_id)],
) -> JSONResponse:
    response, status = request_service.decline_request(
        user_id=user_id,
        request_id=request_id,
    )
    return JSONResponse(content=response, status_code=status)


@router.post("/requests/{request_id}/cancel")
def cancel_request(
    request_id: str,
    user_id: Annotated[str, Depends(current_user_id)],
) -> JSONResponse:
    response, status = request_service.cancel_request(
        user_id=user_id,
        request_id=request_id,
    )
    return JSONResponse(content=response, status_code=status)


__all__ = ["router"]
