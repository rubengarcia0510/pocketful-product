"""Update the main FastAPI app to wire auth + me routes and seed users from /_test/reset."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import bcrypt
from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from . import db as db_mod
from .api.auth import router as auth_router
from .api.me import router as me_router
from .api.payments import router as payments_router
from .api.requests import router as requests_router
from .authorizations import repository as authz_repo
from .errors import error_response, http_exc_to_response


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_mod.get_connection()
    yield


app = FastAPI(title="Pocketful Stage 1", lifespan=lifespan)
app.include_router(auth_router)
app.include_router(me_router)
app.include_router(payments_router)
app.include_router(requests_router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return error_response(422, "validation_failed", _validation_message(exc))


from fastapi import HTTPException as _HTTPException


@app.exception_handler(_HTTPException)
async def http_exception_handler(request: Request, exc: _HTTPException):
    return http_exc_to_response(exc, request)


def _validation_message(exc: RequestValidationError) -> str:
    errs = exc.errors()
    if not errs:
        return "validation failed"
    first = errs[0]
    loc = ".".join(str(x) for x in first.get("loc", ()))
    msg = first.get("msg", "validation failed")
    return f"{loc}: {msg}" if loc else msg


@app.middleware("http")
async def errors_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        from fastapi import HTTPException

        if isinstance(exc, HTTPException):
            return http_exc_to_response(exc, request)
        return error_response(500, "internal_error", "internal error")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/_test/reset", status_code=204)
def test_reset(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
    """Replace service state with the supplied fixture body (spec §3.3, §4).

    ST-1.2 implementation: parses the fixture, seeds users with their
    starting balance. Currency + minor_units are taken from the fixture and
    apply to the whole service. ST-1.3+ will extend this to seed payments
    and requests from the same payload.

    Stage-2: also seeds authorisations (open holds) from the fixture's
    ``authorizations`` array and validates that the sum of seeded open
    holds per payer does not exceed that payer's seeded balance.
    """
    from .authorizations.repository import AuthorizationSeedError

    currency = str(payload.get("currency", "EUR"))
    minor_units = int(payload.get("minor_units", 2))

    if minor_units not in (0, 2, 3):
        from .errors import raise_error

        raise_error(422, "validation_failed", "minor_units must be 0, 2 or 3")

    raw_ttl = payload.get("authorization_ttl_seconds", 600)
    if isinstance(raw_ttl, bool) or not isinstance(raw_ttl, int) or raw_ttl < 1:
        from .errors import raise_error

        raise_error(
            422,
            "validation_failed",
            "authorization_ttl_seconds must be a positive integer",
        )
    authorization_ttl_seconds = raw_ttl

    conn = db_mod.get_connection()
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DELETE FROM authorization_captures")
        conn.execute("DELETE FROM authorizations")
        conn.execute("DELETE FROM tokens")
        conn.execute("DELETE FROM split_participants")
        conn.execute("DELETE FROM splits")
        conn.execute("DELETE FROM settlements")
        conn.execute("DELETE FROM payment_requests")
        conn.execute("DELETE FROM payments")
        conn.execute("DELETE FROM idempotency_keys")
        conn.execute("DELETE FROM users")
        conn.execute(
            "DELETE FROM service_meta WHERE key IN "
            "('currency','minor_units','authorization_ttl_seconds')"
        )

        conn.execute(
            "INSERT INTO service_meta(key, value) VALUES('currency', ?)",
            (currency,),
        )
        conn.execute(
            "INSERT INTO service_meta(key, value) VALUES('minor_units', ?)",
            (str(minor_units),),
        )
        conn.execute(
            "INSERT INTO service_meta(key, value) VALUES"
            "('authorization_ttl_seconds', ?)",
            (str(authorization_ttl_seconds),),
        )

        for user in payload.get("users", []):
            balance = int(user.get("balance", 0))
            if balance < 0:
                from .errors import raise_error

                raise_error(
                    422, "validation_failed", "balance below zero in fixture"
                )
            password = user.get("password", "")
            if not isinstance(password, str) or len(password) < 8:
                from .errors import raise_error

                raise_error(
                    422,
                    "validation_failed",
                    "seeded user password must be at least 8 characters",
                )
            password_hash = bcrypt.hashpw(
                password.encode("utf-8"), bcrypt.gensalt()
            ).decode("ascii")
            conn.execute(
                "INSERT INTO users(id, email, password_hash, display_name, handle, balance, currency, minor_units, created_at) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(user.get("id")),
                    str(user.get("email")),
                    password_hash,
                    str(user.get("display_name", "")),
                    str(user.get("handle")),
                    balance,
                    currency,
                    minor_units,
                    _utc_now_iso(),
                ),
            )

        raw_authorizations = payload.get("authorizations", []) or []
        try:
            authz_repo.validate_seed_authorizations(
                conn, raw_authorizations, now_iso=_utc_now_iso()
            )
        except AuthorizationSeedError as exc:
            from .errors import raise_error

            raise_error(422, "validation_failed", str(exc))

        for raw in raw_authorizations:
            authz_repo.seed_authorization(
                conn,
                authorization_id=str(raw["id"]),
                from_user_id=str(raw["from_user_id"]),
                to_user_id=str(raw["to_user_id"]),
                amount=int(raw["amount"]),
                note=str(raw.get("note", "")),
                visibility=str(raw.get("visibility", "public")),
                status=str(raw.get("status", "open")),
                expires_at=str(raw["expires_at"]),
                created_at=_utc_now_iso(),
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return Response(status_code=204)


@app.get("/_test/export")
def test_export() -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={"error": {"code": "not_implemented", "message": "export arrives with ST-1.3"}},
    )


@app.post("/_test/import", status_code=204)
def test_import(payload: dict[str, Any] = Body(default_factory=dict)) -> Response:
    return Response(status_code=501)


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = ["app"]
