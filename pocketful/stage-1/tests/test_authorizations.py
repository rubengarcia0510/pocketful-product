"""Stage-2 authorisations HTTP API tests (PDF-36).

Tests cover:
  * POST /authorizations — create a hold
  * POST /authorizations/{id}/captures — capture against a hold

Both require Idempotency-Key header per spec §7.
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.asyncio


async def test_create_authorization_success(client, fresh_db):
    """1. Valid authenticated request returns 201 with spec response."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "authorization_ttl_seconds": 600,
        "users": [
            {
                "id": "u_payer",
                "email": "payer@example.com",
                "password": "correct horse",
                "display_name": "Payer",
                "handle": "payer",
                "balance": 10000,
            },
            {
                "id": "u_receiver",
                "email": "receiver@example.com",
                "password": "correct horse",
                "display_name": "Receiver",
                "handle": "receiver",
                "balance": 0,
            },
        ],
        "authorizations": [],
    }
    r = await client.post("/_test/reset", json=fixture)
    assert r.status_code == 204

    login = await client.post(
        "/auth/login",
        json={"email": "payer@example.com", "password": "correct horse"},
    )
    assert login.status_code == 200
    token = login.json()["token"]

    r = await client.post(
        "/authorizations",
        json={"to_handle": "receiver", "amount": 1500, "note": "dinner", "visibility": "public"},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "auth-key-001"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["from_user_id"] == "u_payer"
    assert body["from_handle"] == "payer"
    assert body["to_user_id"] == "u_receiver"
    assert body["to_handle"] == "receiver"
    assert body["amount"] == 1500
    assert body["currency"] == "EUR"
    assert body["captured_amount"] == 0
    assert body["status"] == "open"
    assert body["note"] == "dinner"
    assert body["visibility"] == "public"
    assert "authorization_id" in body
    assert "expires_at" in body
    assert "created_at" in body


async def test_authorization_idempotency_replay_same_body(client, fresh_db):
    """2. Replay with same key and same body returns 200, no duplicate."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "authorization_ttl_seconds": 600,
        "users": [
            {
                "id": "u_payer",
                "email": "payer@example.com",
                "password": "correct horse",
                "display_name": "Payer",
                "handle": "payer",
                "balance": 10000,
            },
            {
                "id": "u_receiver",
                "email": "receiver@example.com",
                "password": "correct horse",
                "display_name": "Receiver",
                "handle": "receiver",
                "balance": 0,
            },
        ],
        "authorizations": [],
    }
    await client.post("/_test/reset", json=fixture)

    login = await client.post(
        "/auth/login",
        json={"email": "payer@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r1 = await client.post(
        "/authorizations",
        json={"to_handle": "receiver", "amount": 500, "note": "coffee", "visibility": "public"},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "auth-replay"},
    )
    assert r1.status_code == 201

    r2 = await client.post(
        "/authorizations",
        json={"to_handle": "receiver", "amount": 500, "note": "coffee", "visibility": "public"},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "auth-replay"},
    )
    assert r2.status_code == 200
    assert r1.json() == r2.json()

    from app.db import get_connection

    conn = get_connection()
    n = conn.execute("SELECT COUNT(*) AS c FROM authorizations").fetchone()["c"]
    assert n == 1, "authorization should not be duplicated"


async def test_authorization_idempotency_different_body(client, fresh_db):
    """3. Same key, different body returns 409 idempotency_key_reuse."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "authorization_ttl_seconds": 600,
        "users": [
            {
                "id": "u_payer",
                "email": "payer@example.com",
                "password": "correct horse",
                "display_name": "Payer",
                "handle": "payer",
                "balance": 10000,
            },
            {
                "id": "u_receiver",
                "email": "receiver@example.com",
                "password": "correct horse",
                "display_name": "Receiver",
                "handle": "receiver",
                "balance": 0,
            },
        ],
        "authorizations": [],
    }
    await client.post("/_test/reset", json=fixture)

    login = await client.post(
        "/auth/login",
        json={"email": "payer@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r1 = await client.post(
        "/authorizations",
        json={"to_handle": "receiver", "amount": 500, "note": "coffee"},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "auth-body-test"},
    )
    assert r1.status_code == 201

    r2 = await client.post(
        "/authorizations",
        json={"to_handle": "receiver", "amount": 600, "note": "different"},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "auth-body-test"},
    )
    assert r2.status_code == 409
    assert r2.json()["error"]["code"] == "idempotency_key_reuse"


async def test_authorization_missing_idempotency_key(client, fresh_db):
    """4. Missing Idempotency-Key returns 400 missing_idempotency_key."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "authorization_ttl_seconds": 600,
        "users": [
            {
                "id": "u_payer",
                "email": "payer@example.com",
                "password": "correct horse",
                "display_name": "Payer",
                "handle": "payer",
                "balance": 10000,
            },
            {
                "id": "u_receiver",
                "email": "receiver@example.com",
                "password": "correct horse",
                "display_name": "Receiver",
                "handle": "receiver",
                "balance": 0,
            },
        ],
        "authorizations": [],
    }
    await client.post("/_test/reset", json=fixture)

    login = await client.post(
        "/auth/login",
        json={"email": "payer@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.post(
        "/authorizations",
        json={"to_handle": "receiver", "amount": 500},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "missing_idempotency_key"


async def test_authorization_unauthenticated(client, fresh_db):
    """5. No token returns 401."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "authorization_ttl_seconds": 600,
        "users": [
            {
                "id": "u_payer",
                "email": "payer@example.com",
                "password": "correct horse",
                "display_name": "Payer",
                "handle": "payer",
                "balance": 10000,
            },
            {
                "id": "u_receiver",
                "email": "receiver@example.com",
                "password": "correct horse",
                "display_name": "Receiver",
                "handle": "receiver",
                "balance": 0,
            },
        ],
        "authorizations": [],
    }
    await client.post("/_test/reset", json=fixture)

    r = await client.post(
        "/authorizations",
        json={"to_handle": "receiver", "amount": 500},
        headers={"Idempotency-Key": "auth-no-auth"},
    )
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthenticated"


async def test_authorization_insufficient_available_funds(client, fresh_db):
    """6. Insufficient available funds returns 409 insufficient_funds."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "authorization_ttl_seconds": 600,
        "users": [
            {
                "id": "u_payer",
                "email": "payer@example.com",
                "password": "correct horse",
                "display_name": "Payer",
                "handle": "payer",
                "balance": 100,
            },
            {
                "id": "u_receiver",
                "email": "receiver@example.com",
                "password": "correct horse",
                "display_name": "Receiver",
                "handle": "receiver",
                "balance": 0,
            },
        ],
        "authorizations": [],
    }
    await client.post("/_test/reset", json=fixture)

    login = await client.post(
        "/auth/login",
        json={"email": "payer@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.post(
        "/authorizations",
        json={"to_handle": "receiver", "amount": 500},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "auth-no-funds"},
    )
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "insufficient_funds"


async def test_authorization_self_payment(client, fresh_db):
    """7. Self-payment returns 422 self_payment."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "authorization_ttl_seconds": 600,
        "users": [
            {
                "id": "u_payer",
                "email": "payer@example.com",
                "password": "correct horse",
                "display_name": "Payer",
                "handle": "payer",
                "balance": 10000,
            },
        ],
        "authorizations": [],
    }
    await client.post("/_test/reset", json=fixture)

    login = await client.post(
        "/auth/login",
        json={"email": "payer@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.post(
        "/authorizations",
        json={"to_handle": "payer", "amount": 500},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "auth-self"},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "self_payment"


async def test_authorization_recipient_not_found(client, fresh_db):
    """8. Non-existent recipient returns 404."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "authorization_ttl_seconds": 600,
        "users": [
            {
                "id": "u_payer",
                "email": "payer@example.com",
                "password": "correct horse",
                "display_name": "Payer",
                "handle": "payer",
                "balance": 10000,
            },
        ],
        "authorizations": [],
    }
    await client.post("/_test/reset", json=fixture)

    login = await client.post(
        "/auth/login",
        json={"email": "payer@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.post(
        "/authorizations",
        json={"to_handle": "nobody", "amount": 500},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "auth-404"},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


async def test_capture_authorization_success(client, fresh_db):
    """9. Capture returns 201 with payment response."""
    from datetime import datetime, timezone

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    expires_iso = (datetime.now(timezone.utc).replace(hour=23, minute=59, second=59) + timezone.utc.utcoffset(None) or datetime.now(timezone.utc)).isoformat(timespec="seconds")

    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "authorization_ttl_seconds": 600,
        "users": [
            {
                "id": "u_payer",
                "email": "payer@example.com",
                "password": "correct horse",
                "display_name": "Payer",
                "handle": "payer",
                "balance": 10000,
            },
            {
                "id": "u_receiver",
                "email": "receiver@example.com",
                "password": "correct horse",
                "display_name": "Receiver",
                "handle": "receiver",
                "balance": 0,
            },
        ],
        "authorizations": [
            {
                "id": "auth_001",
                "from_user_id": "u_payer",
                "to_user_id": "u_receiver",
                "amount": 5000,
                "note": "test",
                "visibility": "public",
                "status": "open",
                "expires_at": "2099-12-31T23:59:59+00:00",
            },
        ],
    }
    r = await client.post("/_test/reset", json=fixture)
    assert r.status_code == 204

    login = await client.post(
        "/auth/login",
        json={"email": "receiver@example.com", "password": "correct horse"},
    )
    assert login.status_code == 200
    token = login.json()["token"]

    r = await client.post(
        "/authorizations/auth_001/captures",
        json={"amount": 1500, "final": True},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "cap-key-001"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["payment_id"].startswith("p_")
    assert body["from_user_id"] == "u_payer"
    assert body["to_user_id"] == "u_receiver"
    assert body["amount"] == 1500
    assert body["currency"] == "EUR"
    assert body["authorization_id"] == "auth_001"


async def test_capture_authorization_idempotency_replay(client, fresh_db):
    """10. Capture replay with same key returns 200."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "authorization_ttl_seconds": 600,
        "users": [
            {
                "id": "u_payer",
                "email": "payer@example.com",
                "password": "correct horse",
                "display_name": "Payer",
                "handle": "payer",
                "balance": 10000,
            },
            {
                "id": "u_receiver",
                "email": "receiver@example.com",
                "password": "correct horse",
                "display_name": "Receiver",
                "handle": "receiver",
                "balance": 0,
            },
        ],
        "authorizations": [
            {
                "id": "auth_002",
                "from_user_id": "u_payer",
                "to_user_id": "u_receiver",
                "amount": 5000,
                "note": "test",
                "visibility": "public",
                "status": "open",
                "expires_at": "2099-12-31T23:59:59+00:00",
            },
        ],
    }
    await client.post("/_test/reset", json=fixture)

    login = await client.post(
        "/auth/login",
        json={"email": "receiver@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r1 = await client.post(
        "/authorizations/auth_002/captures",
        json={"amount": 1000, "final": True},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "cap-replay"},
    )
    assert r1.status_code == 201

    r2 = await client.post(
        "/authorizations/auth_002/captures",
        json={"amount": 1000, "final": True},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "cap-replay"},
    )
    assert r2.status_code == 200
    assert r1.json() == r2.json()


async def test_capture_authorization_idempotency_conflict(client, fresh_db):
    """11. Capture with same key, different body returns 409."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "authorization_ttl_seconds": 600,
        "users": [
            {
                "id": "u_payer",
                "email": "payer@example.com",
                "password": "correct horse",
                "display_name": "Payer",
                "handle": "payer",
                "balance": 10000,
            },
            {
                "id": "u_receiver",
                "email": "receiver@example.com",
                "password": "correct horse",
                "display_name": "Receiver",
                "handle": "receiver",
                "balance": 0,
            },
        ],
        "authorizations": [
            {
                "id": "auth_003",
                "from_user_id": "u_payer",
                "to_user_id": "u_receiver",
                "amount": 5000,
                "note": "test",
                "visibility": "public",
                "status": "open",
                "expires_at": "2099-12-31T23:59:59+00:00",
            },
        ],
    }
    await client.post("/_test/reset", json=fixture)

    login = await client.post(
        "/auth/login",
        json={"email": "receiver@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r1 = await client.post(
        "/authorizations/auth_003/captures",
        json={"amount": 1000},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "cap-conflict"},
    )
    assert r1.status_code == 201

    r2 = await client.post(
        "/authorizations/auth_003/captures",
        json={"amount": 2000},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "cap-conflict"},
    )
    assert r2.status_code == 409
    assert r2.json()["error"]["code"] == "idempotency_key_reuse"


async def test_capture_authorization_missing_key(client, fresh_db):
    """12. Capture without Idempotency-Key returns 400."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "authorization_ttl_seconds": 600,
        "users": [
            {
                "id": "u_payer",
                "email": "payer@example.com",
                "password": "correct horse",
                "display_name": "Payer",
                "handle": "payer",
                "balance": 10000,
            },
            {
                "id": "u_receiver",
                "email": "receiver@example.com",
                "password": "correct horse",
                "display_name": "Receiver",
                "handle": "receiver",
                "balance": 0,
            },
        ],
        "authorizations": [
            {
                "id": "auth_004",
                "from_user_id": "u_payer",
                "to_user_id": "u_receiver",
                "amount": 5000,
                "note": "test",
                "visibility": "public",
                "status": "open",
                "expires_at": "2099-12-31T23:59:59+00:00",
            },
        ],
    }
    await client.post("/_test/reset", json=fixture)

    login = await client.post(
        "/auth/login",
        json={"email": "receiver@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.post(
        "/authorizations/auth_004/captures",
        json={"amount": 1000},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "missing_idempotency_key"


async def test_capture_only_receiver_can_capture(client, fresh_db):
    """13. Only the receiver (to_user) can capture."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "authorization_ttl_seconds": 600,
        "users": [
            {
                "id": "u_payer",
                "email": "payer@example.com",
                "password": "correct horse",
                "display_name": "Payer",
                "handle": "payer",
                "balance": 10000,
            },
            {
                "id": "u_receiver",
                "email": "receiver@example.com",
                "password": "correct horse",
                "display_name": "Receiver",
                "handle": "receiver",
                "balance": 0,
            },
            {
                "id": "u_third",
                "email": "third@example.com",
                "password": "correct horse",
                "display_name": "Third",
                "handle": "third",
                "balance": 0,
            },
        ],
        "authorizations": [
            {
                "id": "auth_005",
                "from_user_id": "u_payer",
                "to_user_id": "u_receiver",
                "amount": 5000,
                "note": "test",
                "visibility": "public",
                "status": "open",
                "expires_at": "2099-12-31T23:59:59+00:00",
            },
        ],
    }
    await client.post("/_test/reset", json=fixture)

    login = await client.post(
        "/auth/login",
        json={"email": "payer@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.post(
        "/authorizations/auth_005/captures",
        json={"amount": 1000},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "cap-forbidden"},
    )
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "forbidden"


async def test_capture_not_open(client, fresh_db):
    """14. Cannot capture a non-open authorization."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "authorization_ttl_seconds": 600,
        "users": [
            {
                "id": "u_payer",
                "email": "payer@example.com",
                "password": "correct horse",
                "display_name": "Payer",
                "handle": "payer",
                "balance": 10000,
            },
            {
                "id": "u_receiver",
                "email": "receiver@example.com",
                "password": "correct horse",
                "display_name": "Receiver",
                "handle": "receiver",
                "balance": 0,
            },
        ],
        "authorizations": [
            {
                "id": "auth_006",
                "from_user_id": "u_payer",
                "to_user_id": "u_receiver",
                "amount": 5000,
                "note": "test",
                "visibility": "public",
                "status": "captured",
                "expires_at": "2099-12-31T23:59:59+00:00",
            },
        ],
    }
    await client.post("/_test/reset", json=fixture)

    login = await client.post(
        "/auth/login",
        json={"email": "receiver@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.post(
        "/authorizations/auth_006/captures",
        json={"amount": 1000},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "cap-not-open"},
    )
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "authorization_not_open"


async def test_capture_exceeds_remaining(client, fresh_db):
    """15. Capture amount exceeds remaining returns 422."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "authorization_ttl_seconds": 600,
        "users": [
            {
                "id": "u_payer",
                "email": "payer@example.com",
                "password": "correct horse",
                "display_name": "Payer",
                "handle": "payer",
                "balance": 10000,
            },
            {
                "id": "u_receiver",
                "email": "receiver@example.com",
                "password": "correct horse",
                "display_name": "Receiver",
                "handle": "receiver",
                "balance": 0,
            },
        ],
        "authorizations": [
            {
                "id": "auth_007",
                "from_user_id": "u_payer",
                "to_user_id": "u_receiver",
                "amount": 1000,
                "note": "test",
                "visibility": "public",
                "status": "open",
                "expires_at": "2099-12-31T23:59:59+00:00",
            },
        ],
    }
    await client.post("/_test/reset", json=fixture)

    login = await client.post(
        "/auth/login",
        json={"email": "receiver@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.post(
        "/authorizations/auth_007/captures",
        json={"amount": 2000},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "cap-exceeds"},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "capture_exceeds_authorization"
