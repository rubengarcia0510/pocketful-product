"""POST /requests tests (spec §8)."""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.asyncio


async def test_create_request_success(client, fresh_db):
    """Valid authenticated request returns 201 with exact spec response."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {
                "id": "u_ada",
                "email": "ada@example.com",
                "password": "correct horse",
                "display_name": "Ada",
                "handle": "ada",
                "balance": 10000,
            },
            {
                "id": "u_bob",
                "email": "bob@example.com",
                "password": "correct horse",
                "display_name": "Bob",
                "handle": "bob",
                "balance": 0,
            },
        ],
        "payments": [],
        "requests": [],
    }
    r = await client.post("/_test/reset", json=fixture)
    assert r.status_code == 204

    login = await client.post(
        "/auth/login",
        json={"email": "bob@example.com", "password": "correct horse"},
    )
    assert login.status_code == 200
    token = login.json()["token"]

    r = await client.post(
        "/requests",
        json={"payer_handle": "ada", "amount": 1200, "note": "taxi"},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "req-key-001"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["requester_id"] == "u_bob"
    assert body["requester_handle"] == "bob"
    assert body["payer_id"] == "u_ada"
    assert body["payer_handle"] == "ada"
    assert body["amount"] == 1200
    assert body["currency"] == "EUR"
    assert body["note"] == "taxi"
    assert body["status"] == "pending"
    assert body["payment_id"] is None
    assert "request_id" in body
    assert "created_at" in body


async def test_request_idempotency_replay_same_body(client, fresh_db):
    """Replay with same key and same body returns 200, no duplicate request."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {
                "id": "u_ada",
                "email": "ada@example.com",
                "password": "correct horse",
                "display_name": "Ada",
                "handle": "ada",
                "balance": 10000,
            },
            {
                "id": "u_bob",
                "email": "bob@example.com",
                "password": "correct horse",
                "display_name": "Bob",
                "handle": "bob",
                "balance": 0,
            },
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login = await client.post(
        "/auth/login",
        json={"email": "bob@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r1 = await client.post(
        "/requests",
        json={"payer_handle": "ada", "amount": 500, "note": "coffee"},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "req-replay"},
    )
    assert r1.status_code == 201

    r2 = await client.post(
        "/requests",
        json={"payer_handle": "ada", "amount": 500, "note": "coffee"},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "req-replay"},
    )
    assert r2.status_code == 200
    assert r1.json() == r2.json()

    from app.db import get_connection

    conn = get_connection()
    n = conn.execute("SELECT COUNT(*) AS c FROM payment_requests").fetchone()["c"]
    assert n == 1, "request should not be duplicated"


async def test_request_idempotency_different_body(client, fresh_db):
    """Same key, different body returns 409 idempotency_key_reuse."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {
                "id": "u_ada",
                "email": "ada@example.com",
                "password": "correct horse",
                "display_name": "Ada",
                "handle": "ada",
                "balance": 10000,
            },
            {
                "id": "u_bob",
                "email": "bob@example.com",
                "password": "correct horse",
                "display_name": "Bob",
                "handle": "bob",
                "balance": 0,
            },
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login = await client.post(
        "/auth/login",
        json={"email": "bob@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r1 = await client.post(
        "/requests",
        json={"payer_handle": "ada", "amount": 500, "note": "coffee"},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "req-body-test"},
    )
    assert r1.status_code == 201

    r2 = await client.post(
        "/requests",
        json={"payer_handle": "ada", "amount": 600, "note": "different"},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "req-body-test"},
    )
    assert r2.status_code == 409
    assert r2.json()["error"]["code"] == "idempotency_key_reuse"


async def test_request_validation_self_request(client, fresh_db):
    """Self-request returns 422 self_request."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {
                "id": "u_ada",
                "email": "ada@example.com",
                "password": "correct horse",
                "display_name": "Ada",
                "handle": "ada",
                "balance": 10000,
            },
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login = await client.post(
        "/auth/login",
        json={"email": "ada@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.post(
        "/requests",
        json={"payer_handle": "ada", "amount": 500},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "req-self"},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "self_request"


async def test_request_validation_invalid_amount(client, fresh_db):
    """Invalid amount returns 422 validation_failed."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {
                "id": "u_ada",
                "email": "ada@example.com",
                "password": "correct horse",
                "display_name": "Ada",
                "handle": "ada",
                "balance": 10000,
            },
            {
                "id": "u_bob",
                "email": "bob@example.com",
                "password": "correct horse",
                "display_name": "Bob",
                "handle": "bob",
                "balance": 0,
            },
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login = await client.post(
        "/auth/login",
        json={"email": "bob@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.post(
        "/requests",
        json={"payer_handle": "ada", "amount": -1},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "req-neg"},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_failed"


async def test_request_validation_note_too_long(client, fresh_db):
    """Note > 200 chars returns 422 validation_failed."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {
                "id": "u_ada",
                "email": "ada@example.com",
                "password": "correct horse",
                "display_name": "Ada",
                "handle": "ada",
                "balance": 10000,
            },
            {
                "id": "u_bob",
                "email": "bob@example.com",
                "password": "correct horse",
                "display_name": "Bob",
                "handle": "bob",
                "balance": 0,
            },
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login = await client.post(
        "/auth/login",
        json={"email": "bob@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.post(
        "/requests",
        json={"payer_handle": "ada", "amount": 500, "note": "x" * 201},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "req-note"},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_failed"


async def test_request_validation_recipient_not_found(client, fresh_db):
    """Unknown payer returns 404 not_found."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {
                "id": "u_ada",
                "email": "ada@example.com",
                "password": "correct horse",
                "display_name": "Ada",
                "handle": "ada",
                "balance": 10000,
            },
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login = await client.post(
        "/auth/login",
        json={"email": "ada@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.post(
        "/requests",
        json={"payer_handle": "nobody", "amount": 500},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "req-404"},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


async def test_request_missing_idempotency_key(client, fresh_db):
    """Missing Idempotency-Key returns 400 missing_idempotency_key."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {
                "id": "u_ada",
                "email": "ada@example.com",
                "password": "correct horse",
                "display_name": "Ada",
                "handle": "ada",
                "balance": 10000,
            },
            {
                "id": "u_bob",
                "email": "bob@example.com",
                "password": "correct horse",
                "display_name": "Bob",
                "handle": "bob",
                "balance": 0,
            },
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login = await client.post(
        "/auth/login",
        json={"email": "bob@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r = await client.post(
        "/requests",
        json={"payer_handle": "ada", "amount": 500},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "missing_idempotency_key"


async def test_request_unauthenticated(client, fresh_db):
    """No token returns 401."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {
                "id": "u_ada",
                "email": "ada@example.com",
                "password": "correct horse",
                "display_name": "Ada",
                "handle": "ada",
                "balance": 10000,
            },
            {
                "id": "u_bob",
                "email": "bob@example.com",
                "password": "correct horse",
                "display_name": "Bob",
                "handle": "bob",
                "balance": 0,
            },
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    r = await client.post(
        "/requests",
        json={"payer_handle": "ada", "amount": 500},
        headers={"Idempotency-Key": "req-no-auth"},
    )
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthenticated"


async def test_list_requests_success(client, fresh_db):
    """GET /requests returns 200 with correct shape."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {"id": "u_ada", "email": "ada@example.com", "password": "correct horse",
             "display_name": "Ada", "handle": "ada", "balance": 10000},
            {"id": "u_bob", "email": "bob@example.com", "password": "correct horse",
             "display_name": "Bob", "handle": "bob", "balance": 0},
            {"id": "u_cy", "email": "cy@example.com", "password": "correct horse",
             "display_name": "Cy", "handle": "cy", "balance": 0},
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login = await client.post("/auth/login", json={"email": "bob@example.com", "password": "correct horse"})
    token = login.json()["token"]

    await client.post("/requests",
        json={"payer_handle": "ada", "amount": 500},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "r1"})
    await client.post("/requests",
        json={"payer_handle": "cy", "amount": 300},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "r2"})

    r = await client.get("/requests", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    body = r.json()
    assert "requests" in body
    assert "has_more" in body
    assert len(body["requests"]) == 2


async def test_list_requests_unauthenticated(client, fresh_db):
    """GET /requests without token returns 401."""
    r = await client.get("/requests")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthenticated"


async def test_list_requests_direction_incoming(client, fresh_db):
    """GET /requests?direction=incoming returns only requests where caller is payer."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {"id": "u_ada", "email": "ada@example.com", "password": "correct horse",
             "display_name": "Ada", "handle": "ada", "balance": 10000},
            {"id": "u_bob", "email": "bob@example.com", "password": "correct horse",
             "display_name": "Bob", "handle": "bob", "balance": 0},
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login_bob = await client.post("/auth/login", json={"email": "bob@example.com", "password": "correct horse"})
    token_bob = login_bob.json()["token"]

    login_ada = await client.post("/auth/login", json={"email": "ada@example.com", "password": "correct horse"})
    token_ada = login_ada.json()["token"]

    await client.post("/requests",
        json={"payer_handle": "bob", "amount": 500},
        headers={"Authorization": f"Bearer {token_ada}", "Idempotency-Key": "inc1"})

    r = await client.get("/requests?direction=incoming", headers={"Authorization": f"Bearer {token_bob}"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["requests"]) == 1
    assert body["requests"][0]["payer_handle"] == "bob"


async def test_pay_request_success(client, fresh_db):
    """Payer pays a pending request, returns 201 with payment shape."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {"id": "u_ada", "email": "ada@example.com", "password": "correct horse",
             "display_name": "Ada", "handle": "ada", "balance": 10000},
            {"id": "u_bob", "email": "bob@example.com", "password": "correct horse",
             "display_name": "Bob", "handle": "bob", "balance": 0},
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login_bob = await client.post("/auth/login", json={"email": "bob@example.com", "password": "correct horse"})
    token_bob = login_bob.json()["token"]

    login_ada = await client.post("/auth/login", json={"email": "ada@example.com", "password": "correct horse"})
    token_ada = login_ada.json()["token"]

    r = await client.post("/requests",
        json={"payer_handle": "ada", "amount": 500},
        headers={"Authorization": f"Bearer {token_bob}", "Idempotency-Key": "create-req-1"})
    assert r.status_code == 201
    request_id = r.json()["request_id"]

    r = await client.post(f"/requests/{request_id}/pay",
        json={"visibility": "public"},
        headers={"Authorization": f"Bearer {token_ada}", "Idempotency-Key": "pay-key-1"})
    assert r.status_code == 201
    body = r.json()
    assert body["payment_id"] is not None
    assert body["from_user_id"] == "u_ada"
    assert body["to_user_id"] == "u_bob"
    assert body["amount"] == 500
    assert body["request_id"] == request_id
    assert body["visibility"] == "public"

    r = await client.get("/requests", headers={"Authorization": f"Bearer {token_bob}"})
    req = r.json()["requests"][0]
    assert req["status"] == "paid"
    assert req["payment_id"] == body["payment_id"]


async def test_pay_request_replay(client, fresh_db):
    """Replaying a successful pay returns 200 with same response."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {"id": "u_ada", "email": "ada@example.com", "password": "correct horse",
             "display_name": "Ada", "handle": "ada", "balance": 10000},
            {"id": "u_bob", "email": "bob@example.com", "password": "correct horse",
             "display_name": "Bob", "handle": "bob", "balance": 0},
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login_bob = await client.post("/auth/login", json={"email": "bob@example.com", "password": "correct horse"})
    token_bob = login_bob.json()["token"]

    login_ada = await client.post("/auth/login", json={"email": "ada@example.com", "password": "correct horse"})
    token_ada = login_ada.json()["token"]

    r = await client.post("/requests",
        json={"payer_handle": "ada", "amount": 500},
        headers={"Authorization": f"Bearer {token_bob}", "Idempotency-Key": "req-replay-pay"})
    request_id = r.json()["request_id"]

    r1 = await client.post(f"/requests/{request_id}/pay",
        json={},
        headers={"Authorization": f"Bearer {token_ada}", "Idempotency-Key": "pay-replay"})
    assert r1.status_code == 201
    payment1 = r1.json()["payment_id"]

    r2 = await client.post(f"/requests/{request_id}/pay",
        json={},
        headers={"Authorization": f"Bearer {token_ada}", "Idempotency-Key": "pay-replay"})
    assert r2.status_code == 200
    assert r2.json()["payment_id"] == payment1


async def test_pay_request_idempotency_conflict(client, fresh_db):
    """Same key, different body returns 409 idempotency_key_reuse."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {"id": "u_ada", "email": "ada@example.com", "password": "correct horse",
             "display_name": "Ada", "handle": "ada", "balance": 10000},
            {"id": "u_bob", "email": "bob@example.com", "password": "correct horse",
             "display_name": "Bob", "handle": "bob", "balance": 0},
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login_bob = await client.post("/auth/login", json={"email": "bob@example.com", "password": "correct horse"})
    token_bob = login_bob.json()["token"]

    login_ada = await client.post("/auth/login", json={"email": "ada@example.com", "password": "correct horse"})
    token_ada = login_ada.json()["token"]

    r = await client.post("/requests",
        json={"payer_handle": "ada", "amount": 500},
        headers={"Authorization": f"Bearer {token_bob}", "Idempotency-Key": "req-conflict"})
    request_id = r.json()["request_id"]

    r1 = await client.post(f"/requests/{request_id}/pay",
        json={"visibility": "public"},
        headers={"Authorization": f"Bearer {token_ada}", "Idempotency-Key": "pay-conflict"})
    assert r1.status_code == 201

    r2 = await client.post(f"/requests/{request_id}/pay",
        json={"visibility": "private"},
        headers={"Authorization": f"Bearer {token_ada}", "Idempotency-Key": "pay-conflict"})
    assert r2.status_code == 409
    assert r2.json()["error"]["code"] == "idempotency_key_reuse"


async def test_pay_request_not_payer(client, fresh_db):
    """Non-payer calling pay returns 403 forbidden."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {"id": "u_ada", "email": "ada@example.com", "password": "correct horse",
             "display_name": "Ada", "handle": "ada", "balance": 10000},
            {"id": "u_bob", "email": "bob@example.com", "password": "correct horse",
             "display_name": "Bob", "handle": "bob", "balance": 0},
            {"id": "u_cy", "email": "cy@example.com", "password": "correct horse",
             "display_name": "Cy", "handle": "cy", "balance": 0},
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login_bob = await client.post("/auth/login", json={"email": "bob@example.com", "password": "correct horse"})
    token_bob = login_bob.json()["token"]

    login_cy = await client.post("/auth/login", json={"email": "cy@example.com", "password": "correct horse"})
    token_cy = login_cy.json()["token"]

    r = await client.post("/requests",
        json={"payer_handle": "ada", "amount": 500},
        headers={"Authorization": f"Bearer {token_bob}", "Idempotency-Key": "req-403"})
    request_id = r.json()["request_id"]

    r = await client.post(f"/requests/{request_id}/pay",
        json={},
        headers={"Authorization": f"Bearer {token_cy}", "Idempotency-Key": "pay-403"})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "forbidden"


async def test_pay_request_not_pending(client, fresh_db):
    """Paying a non-pending request returns 409 request_not_pending."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {"id": "u_ada", "email": "ada@example.com", "password": "correct horse",
             "display_name": "Ada", "handle": "ada", "balance": 10000},
            {"id": "u_bob", "email": "bob@example.com", "password": "correct horse",
             "display_name": "Bob", "handle": "bob", "balance": 0},
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login_bob = await client.post("/auth/login", json={"email": "bob@example.com", "password": "correct horse"})
    token_bob = login_bob.json()["token"]

    login_ada = await client.post("/auth/login", json={"email": "ada@example.com", "password": "correct horse"})
    token_ada = login_ada.json()["token"]

    r = await client.post("/requests",
        json={"payer_handle": "ada", "amount": 500},
        headers={"Authorization": f"Bearer {token_bob}", "Idempotency-Key": "req-409"})
    request_id = r.json()["request_id"]

    await client.post(f"/requests/{request_id}/pay",
        json={},
        headers={"Authorization": f"Bearer {token_ada}", "Idempotency-Key": "pay-409"})

    r = await client.post(f"/requests/{request_id}/pay",
        json={},
        headers={"Authorization": f"Bearer {token_ada}", "Idempotency-Key": "pay-409-again"})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "request_not_pending"


async def test_pay_request_not_found(client, fresh_db):
    """Paying non-existent request returns 404."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {"id": "u_ada", "email": "ada@example.com", "password": "correct horse",
             "display_name": "Ada", "handle": "ada", "balance": 10000},
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login_ada = await client.post("/auth/login", json={"email": "ada@example.com", "password": "correct horse"})
    token = login_ada.json()["token"]

    r = await client.post("/requests/rq_nonexistent/pay",
        json={},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "pay-404"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


async def test_pay_request_insufficient_funds(client, fresh_db):
    """Payer with insufficient available balance returns 409 insufficient_funds."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {"id": "u_ada", "email": "ada@example.com", "password": "correct horse",
             "display_name": "Ada", "handle": "ada", "balance": 100},
            {"id": "u_bob", "email": "bob@example.com", "password": "correct horse",
             "display_name": "Bob", "handle": "bob", "balance": 0},
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login_bob = await client.post("/auth/login", json={"email": "bob@example.com", "password": "correct horse"})
    token_bob = login_bob.json()["token"]

    login_ada = await client.post("/auth/login", json={"email": "ada@example.com", "password": "correct horse"})
    token_ada = login_ada.json()["token"]

    r = await client.post("/requests",
        json={"payer_handle": "ada", "amount": 500},
        headers={"Authorization": f"Bearer {token_bob}", "Idempotency-Key": "req-no-funds"})
    request_id = r.json()["request_id"]

    r = await client.post(f"/requests/{request_id}/pay",
        json={},
        headers={"Authorization": f"Bearer {token_ada}", "Idempotency-Key": "pay-no-funds"})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "insufficient_funds"


async def test_pay_request_missing_idempotency_key(client, fresh_db):
    """Missing Idempotency-Key returns 400."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {"id": "u_ada", "email": "ada@example.com", "password": "correct horse",
             "display_name": "Ada", "handle": "ada", "balance": 10000},
            {"id": "u_bob", "email": "bob@example.com", "password": "correct horse",
             "display_name": "Bob", "handle": "bob", "balance": 0},
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login_bob = await client.post("/auth/login", json={"email": "bob@example.com", "password": "correct horse"})
    token_bob = login_bob.json()["token"]

    login_ada = await client.post("/auth/login", json={"email": "ada@example.com", "password": "correct horse"})
    token_ada = login_ada.json()["token"]

    r = await client.post("/requests",
        json={"payer_handle": "ada", "amount": 500},
        headers={"Authorization": f"Bearer {token_bob}", "Idempotency-Key": "req-no-key"})
    request_id = r.json()["request_id"]

    r = await client.post(f"/requests/{request_id}/pay",
        json={},
        headers={"Authorization": f"Bearer {token_ada}"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "missing_idempotency_key"


async def test_decline_request_success(client, fresh_db):
    """Payer declines a pending request, returns 200 with status declined."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {"id": "u_ada", "email": "ada@example.com", "password": "correct horse",
             "display_name": "Ada", "handle": "ada", "balance": 10000},
            {"id": "u_bob", "email": "bob@example.com", "password": "correct horse",
             "display_name": "Bob", "handle": "bob", "balance": 0},
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login_bob = await client.post("/auth/login", json={"email": "bob@example.com", "password": "correct horse"})
    token_bob = login_bob.json()["token"]

    login_ada = await client.post("/auth/login", json={"email": "ada@example.com", "password": "correct horse"})
    token_ada = login_ada.json()["token"]

    r = await client.post("/requests",
        json={"payer_handle": "ada", "amount": 500},
        headers={"Authorization": f"Bearer {token_bob}", "Idempotency-Key": "decline-req"})
    request_id = r.json()["request_id"]

    r = await client.post(f"/requests/{request_id}/decline",
        headers={"Authorization": f"Bearer {token_ada}"})
    assert r.status_code == 200
    assert r.json()["status"] == "declined"


async def test_decline_request_not_payer(client, fresh_db):
    """Non-payer declining returns 403 forbidden."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {"id": "u_ada", "email": "ada@example.com", "password": "correct horse",
             "display_name": "Ada", "handle": "ada", "balance": 10000},
            {"id": "u_bob", "email": "bob@example.com", "password": "correct horse",
             "display_name": "Bob", "handle": "bob", "balance": 0},
            {"id": "u_cy", "email": "cy@example.com", "password": "correct horse",
             "display_name": "Cy", "handle": "cy", "balance": 0},
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login_bob = await client.post("/auth/login", json={"email": "bob@example.com", "password": "correct horse"})
    token_bob = login_bob.json()["token"]

    login_cy = await client.post("/auth/login", json={"email": "cy@example.com", "password": "correct horse"})
    token_cy = login_cy.json()["token"]

    r = await client.post("/requests",
        json={"payer_handle": "ada", "amount": 500},
        headers={"Authorization": f"Bearer {token_bob}", "Idempotency-Key": "decline-403"})
    request_id = r.json()["request_id"]

    r = await client.post(f"/requests/{request_id}/decline",
        headers={"Authorization": f"Bearer {token_cy}"})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "forbidden"


async def test_decline_request_already_declined(client, fresh_db):
    """Declining an already declined request returns 200 (no-op, empty body)."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {"id": "u_ada", "email": "ada@example.com", "password": "correct horse",
             "display_name": "Ada", "handle": "ada", "balance": 10000},
            {"id": "u_bob", "email": "bob@example.com", "password": "correct horse",
             "display_name": "Bob", "handle": "bob", "balance": 0},
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login_bob = await client.post("/auth/login", json={"email": "bob@example.com", "password": "correct horse"})
    token_bob = login_bob.json()["token"]

    login_ada = await client.post("/auth/login", json={"email": "ada@example.com", "password": "correct horse"})
    token_ada = login_ada.json()["token"]

    r = await client.post("/requests",
        json={"payer_handle": "ada", "amount": 500},
        headers={"Authorization": f"Bearer {token_bob}", "Idempotency-Key": "decline-noop"})
    request_id = r.json()["request_id"]

    r1 = await client.post(f"/requests/{request_id}/decline",
        headers={"Authorization": f"Bearer {token_ada}"})
    assert r1.status_code == 200
    assert r1.json()["status"] == "declined"

    r2 = await client.post(f"/requests/{request_id}/decline",
        headers={"Authorization": f"Bearer {token_ada}"})
    assert r2.status_code == 200
    assert r2.json() == {}


async def test_decline_request_not_pending(client, fresh_db):
    """Declining a non-pending request returns 409."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {"id": "u_ada", "email": "ada@example.com", "password": "correct horse",
             "display_name": "Ada", "handle": "ada", "balance": 10000},
            {"id": "u_bob", "email": "bob@example.com", "password": "correct horse",
             "display_name": "Bob", "handle": "bob", "balance": 0},
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login_bob = await client.post("/auth/login", json={"email": "bob@example.com", "password": "correct horse"})
    token_bob = login_bob.json()["token"]

    login_ada = await client.post("/auth/login", json={"email": "ada@example.com", "password": "correct horse"})
    token_ada = login_ada.json()["token"]

    r = await client.post("/requests",
        json={"payer_handle": "ada", "amount": 500},
        headers={"Authorization": f"Bearer {token_bob}", "Idempotency-Key": "decline-409"})
    request_id = r.json()["request_id"]

    await client.post(f"/requests/{request_id}/pay",
        json={},
        headers={"Authorization": f"Bearer {token_ada}", "Idempotency-Key": "pay-for-decline"})

    r = await client.post(f"/requests/{request_id}/decline",
        headers={"Authorization": f"Bearer {token_ada}"})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "request_not_pending"


async def test_decline_request_not_found(client, fresh_db):
    """Declining non-existent request returns 404."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {"id": "u_ada", "email": "ada@example.com", "password": "correct horse",
             "display_name": "Ada", "handle": "ada", "balance": 10000},
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login_ada = await client.post("/auth/login", json={"email": "ada@example.com", "password": "correct horse"})
    token = login_ada.json()["token"]

    r = await client.post("/requests/rq_nonexistent/decline",
        headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


async def test_cancel_request_success(client, fresh_db):
    """Requester cancels a pending request, returns 200 with status cancelled."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {"id": "u_ada", "email": "ada@example.com", "password": "correct horse",
             "display_name": "Ada", "handle": "ada", "balance": 10000},
            {"id": "u_bob", "email": "bob@example.com", "password": "correct horse",
             "display_name": "Bob", "handle": "bob", "balance": 0},
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login_bob = await client.post("/auth/login", json={"email": "bob@example.com", "password": "correct horse"})
    token_bob = login_bob.json()["token"]

    r = await client.post("/requests",
        json={"payer_handle": "ada", "amount": 500},
        headers={"Authorization": f"Bearer {token_bob}", "Idempotency-Key": "cancel-req"})
    request_id = r.json()["request_id"]

    r = await client.post(f"/requests/{request_id}/cancel",
        headers={"Authorization": f"Bearer {token_bob}"})
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"


async def test_cancel_request_not_requester(client, fresh_db):
    """Non-requester cancelling returns 403 forbidden."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {"id": "u_ada", "email": "ada@example.com", "password": "correct horse",
             "display_name": "Ada", "handle": "ada", "balance": 10000},
            {"id": "u_bob", "email": "bob@example.com", "password": "correct horse",
             "display_name": "Bob", "handle": "bob", "balance": 0},
            {"id": "u_cy", "email": "cy@example.com", "password": "correct horse",
             "display_name": "Cy", "handle": "cy", "balance": 0},
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login_bob = await client.post("/auth/login", json={"email": "bob@example.com", "password": "correct horse"})
    token_bob = login_bob.json()["token"]

    login_cy = await client.post("/auth/login", json={"email": "cy@example.com", "password": "correct horse"})
    token_cy = login_cy.json()["token"]

    r = await client.post("/requests",
        json={"payer_handle": "ada", "amount": 500},
        headers={"Authorization": f"Bearer {token_bob}", "Idempotency-Key": "cancel-403"})
    request_id = r.json()["request_id"]

    r = await client.post(f"/requests/{request_id}/cancel",
        headers={"Authorization": f"Bearer {token_cy}"})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "forbidden"


async def test_cancel_request_already_cancelled(client, fresh_db):
    """Cancelling an already cancelled request returns 200 (no-op, empty body)."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {"id": "u_ada", "email": "ada@example.com", "password": "correct horse",
             "display_name": "Ada", "handle": "ada", "balance": 10000},
            {"id": "u_bob", "email": "bob@example.com", "password": "correct horse",
             "display_name": "Bob", "handle": "bob", "balance": 0},
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login_bob = await client.post("/auth/login", json={"email": "bob@example.com", "password": "correct horse"})
    token_bob = login_bob.json()["token"]

    r = await client.post("/requests",
        json={"payer_handle": "ada", "amount": 500},
        headers={"Authorization": f"Bearer {token_bob}", "Idempotency-Key": "cancel-noop"})
    request_id = r.json()["request_id"]

    r1 = await client.post(f"/requests/{request_id}/cancel",
        headers={"Authorization": f"Bearer {token_bob}"})
    assert r1.status_code == 200
    assert r1.json()["status"] == "cancelled"

    r2 = await client.post(f"/requests/{request_id}/cancel",
        headers={"Authorization": f"Bearer {token_bob}"})
    assert r2.status_code == 200
    assert r2.json() == {}


async def test_cancel_request_not_pending(client, fresh_db):
    """Cancelling a non-pending request returns 409."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {"id": "u_ada", "email": "ada@example.com", "password": "correct horse",
             "display_name": "Ada", "handle": "ada", "balance": 10000},
            {"id": "u_bob", "email": "bob@example.com", "password": "correct horse",
             "display_name": "Bob", "handle": "bob", "balance": 0},
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login_bob = await client.post("/auth/login", json={"email": "bob@example.com", "password": "correct horse"})
    token_bob = login_bob.json()["token"]

    login_ada = await client.post("/auth/login", json={"email": "ada@example.com", "password": "correct horse"})
    token_ada = login_ada.json()["token"]

    r = await client.post("/requests",
        json={"payer_handle": "ada", "amount": 500},
        headers={"Authorization": f"Bearer {token_bob}", "Idempotency-Key": "cancel-409"})
    request_id = r.json()["request_id"]

    await client.post(f"/requests/{request_id}/decline",
        headers={"Authorization": f"Bearer {token_ada}"})

    r = await client.post(f"/requests/{request_id}/cancel",
        headers={"Authorization": f"Bearer {token_bob}"})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "request_not_pending"


async def test_cancel_request_not_found(client, fresh_db):
    """Cancelling non-existent request returns 404."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {"id": "u_bob", "email": "bob@example.com", "password": "correct horse",
             "display_name": "Bob", "handle": "bob", "balance": 0},
        ],
        "payments": [],
        "requests": [],
    }
    await client.post("/_test/reset", json=fixture)

    login_bob = await client.post("/auth/login", json={"email": "bob@example.com", "password": "correct horse"})
    token = login_bob.json()["token"]

    r = await client.post("/requests/rq_nonexistent/cancel",
        headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"
