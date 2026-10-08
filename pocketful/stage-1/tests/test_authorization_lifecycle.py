"""Stage-2 authorisations lifecycle tests (PDF-38).

Tests cover:
  * POST /authorizations/{id}/void — void an open hold
  * GET /authorizations — list with filters and pagination
  * GET /me — total/available/held after hold/capture/void
  * Expiration by clock — read shows expired, capture/void rejected
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.asyncio


async def test_void_authorization_success(client, fresh_db):
    """1. Payer voids an open hold → 200 with status voided."""
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
                "id": "auth_void_001",
                "from_user_id": "u_payer",
                "to_user_id": "u_receiver",
                "amount": 2000,
                "note": "test void",
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
        json={"email": "payer@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.post(
        "/authorizations/auth_void_001/void",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["authorization_id"] == "auth_void_001"
    assert body["status"] == "voided"
    assert body["from_handle"] == "payer"
    assert body["to_handle"] == "receiver"


async def test_void_authorization_idempotent(client, fresh_db):
    """2. Voiding an already-voided authorization returns 200 (idempotent)."""
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
                "id": "auth_void_002",
                "from_user_id": "u_payer",
                "to_user_id": "u_receiver",
                "amount": 2000,
                "note": "test void",
                "visibility": "public",
                "status": "voided",
                "expires_at": "2099-12-31T23:59:59+00:00",
            },
        ],
    }
    r = await client.post("/_test/reset", json=fixture)
    assert r.status_code == 204

    login = await client.post(
        "/auth/login",
        json={"email": "payer@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.post(
        "/authorizations/auth_void_002/void",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "voided"


async def test_void_authorization_not_open_captured(client, fresh_db):
    """3. Voiding a captured authorization returns 409 authorization_not_open."""
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
                "id": "auth_void_003",
                "from_user_id": "u_payer",
                "to_user_id": "u_receiver",
                "amount": 2000,
                "captured_amount": 2000,
                "note": "test void",
                "visibility": "public",
                "status": "captured",
                "expires_at": "2099-12-31T23:59:59+00:00",
            },
        ],
    }
    r = await client.post("/_test/reset", json=fixture)
    assert r.status_code == 204

    login = await client.post(
        "/auth/login",
        json={"email": "payer@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.post(
        "/authorizations/auth_void_003/void",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "authorization_not_open"


async def test_void_authorization_forbidden_not_payer(client, fresh_db):
    """4. Only the payer can void; receiver gets 403 forbidden."""
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
                "id": "auth_void_004",
                "from_user_id": "u_payer",
                "to_user_id": "u_receiver",
                "amount": 2000,
                "note": "test void",
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
    token = login.json()["token"]

    r = await client.post(
        "/authorizations/auth_void_004/void",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "forbidden"


async def test_void_authorization_not_found(client, fresh_db):
    """5. Voiding a non-existent authorization returns 404."""
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
    r = await client.post("/_test/reset", json=fixture)
    assert r.status_code == 204

    login = await client.post(
        "/auth/login",
        json={"email": "payer@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.post(
        "/authorizations/nonexistent/void",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


async def test_list_authorizations_all(client, fresh_db):
    """6. GET /authorizations returns all authorisations for caller."""
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
                "balance": 5000,
            },
        ],
        "authorizations": [
            {
                "id": "auth_list_001",
                "from_user_id": "u_payer",
                "to_user_id": "u_receiver",
                "amount": 1000,
                "note": "outgoing",
                "visibility": "public",
                "status": "open",
                "expires_at": "2099-12-31T23:59:59+00:00",
            },
            {
                "id": "auth_list_002",
                "from_user_id": "u_receiver",
                "to_user_id": "u_payer",
                "amount": 500,
                "note": "incoming",
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
        json={"email": "payer@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.get(
        "/authorizations",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "authorizations" in body
    assert len(body["authorizations"]) == 2


async def test_list_authorizations_direction_filter(client, fresh_db):
    """7. GET /authorizations?direction=outgoing filters correctly."""
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
                "balance": 5000,
            },
        ],
        "authorizations": [
            {
                "id": "auth_out",
                "from_user_id": "u_payer",
                "to_user_id": "u_receiver",
                "amount": 1000,
                "note": "outgoing",
                "visibility": "public",
                "status": "open",
                "expires_at": "2099-12-31T23:59:59+00:00",
            },
            {
                "id": "auth_in",
                "from_user_id": "u_receiver",
                "to_user_id": "u_payer",
                "amount": 500,
                "note": "incoming",
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
        json={"email": "payer@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.get(
        "/authorizations?direction=outgoing",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["authorizations"]) == 1
    assert body["authorizations"][0]["authorization_id"] == "auth_out"


async def test_list_authorizations_status_filter(client, fresh_db):
    """8. GET /authorizations?status=open filters correctly."""
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
                "id": "auth_open",
                "from_user_id": "u_payer",
                "to_user_id": "u_receiver",
                "amount": 1000,
                "note": "open",
                "visibility": "public",
                "status": "open",
                "expires_at": "2099-12-31T23:59:59+00:00",
            },
            {
                "id": "auth_captured",
                "from_user_id": "u_payer",
                "to_user_id": "u_receiver",
                "amount": 500,
                "captured_amount": 500,
                "note": "captured",
                "visibility": "public",
                "status": "captured",
                "expires_at": "2099-12-31T23:59:59+00:00",
            },
        ],
    }
    r = await client.post("/_test/reset", json=fixture)
    assert r.status_code == 204

    login = await client.post(
        "/auth/login",
        json={"email": "payer@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.get(
        "/authorizations?status=open",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["authorizations"]) == 1
    assert body["authorizations"][0]["authorization_id"] == "auth_open"


async def test_list_authorizations_pagination(client, fresh_db):
    """9. GET /authorizations returns has_more for pagination."""
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
                "balance": 100000,
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
            {"id": f"auth_{i}", "from_user_id": "u_payer", "to_user_id": "u_receiver",
             "amount": 100, "note": f"note{i}", "visibility": "public",
             "status": "open", "expires_at": "2099-12-31T23:59:59+00:00"}
            for i in range(5)
        ],
    }
    r = await client.post("/_test/reset", json=fixture)
    assert r.status_code == 204

    login = await client.post(
        "/auth/login",
        json={"email": "payer@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.get(
        "/authorizations?limit=3&offset=0",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["authorizations"]) == 3
    assert body["has_more"] is True


async def test_get_me_with_held(client, fresh_db):
    """10. GET /me includes total/available/held after hold."""
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
                "id": "auth_held_001",
                "from_user_id": "u_payer",
                "to_user_id": "u_receiver",
                "amount": 2000,
                "note": "hold",
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
        json={"email": "payer@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.get(
        "/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["balance"] == 10000
    assert body["total"] == 10000
    assert body["held"] == 2000
    assert body["available"] == 8000


async def test_get_me_after_capture(client, fresh_db):
    """11. GET /me reflects captured amount (held reduced on final capture)."""
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
                "id": "auth_cap_001",
                "from_user_id": "u_payer",
                "to_user_id": "u_receiver",
                "amount": 2000,
                "note": "capture test",
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
    token = login.json()["token"]

    r = await client.post(
        "/authorizations/auth_cap_001/capture",
        json={"amount": 2000, "final": True},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "cap-held-test"},
    )
    assert r.status_code == 201

    login_payer = await client.post(
        "/auth/login",
        json={"email": "payer@example.com", "password": "correct horse"},
    )
    token_payer = login_payer.json()["token"]

    r = await client.get(
        "/me",
        headers={"Authorization": f"Bearer {token_payer}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["balance"] == 8000
    assert body["total"] == 8000
    assert body["held"] == 0
    assert body["available"] == 8000


async def test_get_me_after_void(client, fresh_db):
    """12. GET /me reflects void (held released)."""
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
                "id": "auth_void_me_001",
                "from_user_id": "u_payer",
                "to_user_id": "u_receiver",
                "amount": 3000,
                "note": "void test",
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
        json={"email": "payer@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.post(
        "/authorizations/auth_void_me_001/void",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200

    r = await client.get(
        "/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["balance"] == 10000
    assert body["total"] == 10000
    assert body["held"] == 0
    assert body["available"] == 10000


async def test_expiration_by_clock_shows_expired(client, fresh_db):
    """13. Expired authorization shows status=expired via expiry-by-clock."""
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
                "id": "auth_exp_001",
                "from_user_id": "u_payer",
                "to_user_id": "u_receiver",
                "amount": 2000,
                "note": "expired",
                "visibility": "public",
                "status": "open",
                "expires_at": "2020-01-01T00:00:00+00:00",
            },
        ],
    }
    r = await client.post("/_test/reset", json=fixture)
    assert r.status_code == 204

    login = await client.post(
        "/auth/login",
        json={"email": "payer@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.get(
        "/authorizations",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["authorizations"]) == 1
    assert body["authorizations"][0]["status"] == "expired"


async def test_expiration_by_clock_rejects_capture(client, fresh_db):
    """14. Capture on expired authorization returns 409 authorization_expired."""
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
                "id": "auth_exp_cap_001",
                "from_user_id": "u_payer",
                "to_user_id": "u_receiver",
                "amount": 2000,
                "note": "expired capture",
                "visibility": "public",
                "status": "open",
                "expires_at": "2020-01-01T00:00:00+00:00",
            },
        ],
    }
    r = await client.post("/_test/reset", json=fixture)
    assert r.status_code == 204

    login = await client.post(
        "/auth/login",
        json={"email": "receiver@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.post(
        "/authorizations/auth_exp_cap_001/capture",
        json={"amount": 1000},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "cap-expired"},
    )
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "authorization_expired"


async def test_expiration_by_clock_rejects_void(client, fresh_db):
    """15. Void on expired authorization returns 409 authorization_not_open."""
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
                "id": "auth_exp_void_001",
                "from_user_id": "u_payer",
                "to_user_id": "u_receiver",
                "amount": 2000,
                "note": "expired void",
                "visibility": "public",
                "status": "open",
                "expires_at": "2020-01-01T00:00:00+00:00",
            },
        ],
    }
    r = await client.post("/_test/reset", json=fixture)
    assert r.status_code == 204

    login = await client.post(
        "/auth/login",
        json={"email": "payer@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.post(
        "/authorizations/auth_exp_void_001/void",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "authorization_not_open"


async def test_capture_singular_endpoint(client, fresh_db):
    """16. POST /authorizations/{id}/capture (singular) works like /captures."""
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
                "id": "auth_singular_001",
                "from_user_id": "u_payer",
                "to_user_id": "u_receiver",
                "amount": 3000,
                "note": "singular capture",
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
    token = login.json()["token"]

    r = await client.post(
        "/authorizations/auth_singular_001/capture",
        json={"amount": 1500},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "cap-singular"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["payment_id"].startswith("p_")
    assert body["authorization_id"] == "auth_singular_001"
    assert body["amount"] == 1500
