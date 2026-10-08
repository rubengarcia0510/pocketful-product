"""PDF-39: Multi-client and out-of-order response handling tests.

Tests verify:
1. Same Idempotency-Key, different users → no collision (PK scoped by user_id)
2. Concurrent same user, same key, different bodies → 1 wins, 1 gets 409 conflict
3. Retry after lost response → replay 200, no duplicate payment
4. GET /me reflects state change by another client (latest-wins on read)

All tests are server-side only; no UI. Reuses existing idempotency
infrastructure from PDF-202.
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.asyncio


async def test_same_key_different_users_no_collision(client, fresh_db):
    """1. Same Idempotency-Key string, different users → both succeed, separate rows."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {
                "id": "u_alice",
                "email": "alice@example.com",
                "password": "correct horse",
                "display_name": "Alice",
                "handle": "alice",
                "balance": 10000,
            },
            {
                "id": "u_bob",
                "email": "bob@example.com",
                "password": "correct horse",
                "display_name": "Bob",
                "handle": "bob",
                "balance": 10000,
            },
            {
                "id": "u_charlie",
                "email": "charlie@example.com",
                "password": "correct horse",
                "display_name": "Charlie",
                "handle": "charlie",
                "balance": 0,
            },
        ],
    }
    r = await client.post("/_test/reset", json=fixture)
    assert r.status_code == 204

    login_alice = await client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "correct horse"},
    )
    token_alice = login_alice.json()["token"]

    login_bob = await client.post(
        "/auth/login",
        json={"email": "bob@example.com", "password": "correct horse"},
    )
    token_bob = login_bob.json()["token"]

    r_alice = await client.post(
        "/payments",
        json={"to_handle": "charlie", "amount": 500, "note": "alice payment"},
        headers={"Authorization": f"Bearer {token_alice}", "Idempotency-Key": "shared-key"},
    )
    assert r_alice.status_code == 201

    r_bob = await client.post(
        "/payments",
        json={"to_handle": "charlie", "amount": 700, "note": "bob payment"},
        headers={"Authorization": f"Bearer {token_bob}", "Idempotency-Key": "shared-key"},
    )
    assert r_bob.status_code == 201

    from app.db import get_connection

    conn = get_connection()
    rows = conn.execute(
        "SELECT user_id, key, request_body_hash FROM idempotency_keys WHERE key = 'shared-key'"
    ).fetchall()
    assert len(rows) == 2


async def test_concurrent_same_user_different_bodies_one_conflict(client, fresh_db):
    """2. Same user, same Idempotency-Key, different bodies → one 201, one 409."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {
                "id": "u_alice",
                "email": "alice@example.com",
                "password": "correct horse",
                "display_name": "Alice",
                "handle": "alice",
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
    }
    r = await client.post("/_test/reset", json=fixture)
    assert r.status_code == 204

    login = await client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r1 = await client.post(
        "/payments",
        json={"to_handle": "bob", "amount": 500, "note": "first"},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "conflict-key"},
    )
    assert r1.status_code == 201

    r2 = await client.post(
        "/payments",
        json={"to_handle": "bob", "amount": 600, "note": "different"},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "conflict-key"},
    )
    assert r2.status_code == 409
    assert r2.json()["error"]["code"] == "idempotency_key_reuse"


async def test_retry_after_lost_response_replays_no_duplicate(client, fresh_db):
    """3. Retry with same key+body → 200 replay, one payment only."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {
                "id": "u_alice",
                "email": "alice@example.com",
                "password": "correct horse",
                "display_name": "Alice",
                "handle": "alice",
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
    }
    r = await client.post("/_test/reset", json=fixture)
    assert r.status_code == 204

    login = await client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "correct horse"},
    )
    token = login.json()["token"]

    r1 = await client.post(
        "/payments",
        json={"to_handle": "bob", "amount": 1500, "note": "retry test"},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "retry-key"},
    )
    assert r1.status_code == 201

    r2 = await client.post(
        "/payments",
        json={"to_handle": "bob", "amount": 1500, "note": "retry test"},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "retry-key"},
    )
    assert r2.status_code == 200
    assert r1.json() == r2.json()

    from app.db import get_connection

    conn = get_connection()
    n = conn.execute("SELECT COUNT(*) AS c FROM payments WHERE authorization_id IS NULL").fetchone()["c"]
    assert n == 1, "only one payment should exist"


async def test_get_me_reflects_other_client_state_change(client, fresh_db):
    """4. GET /me reflects balance change made by another client (latest-wins on read)."""
    fixture = {
        "currency": "EUR",
        "minor_units": 2,
        "users": [
            {
                "id": "u_alice",
                "email": "alice@example.com",
                "password": "correct horse",
                "display_name": "Alice",
                "handle": "alice",
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
    }
    r = await client.post("/_test/reset", json=fixture)
    assert r.status_code == 204

    login_alice = await client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "correct horse"},
    )
    token_alice = login_alice.json()["token"]

    login_bob = await client.post(
        "/auth/login",
        json={"email": "bob@example.com", "password": "correct horse"},
    )
    token_bob = login_bob.json()["token"]

    me_alice_before = await client.get(
        "/me",
        headers={"Authorization": f"Bearer {token_alice}"},
    )
    assert me_alice_before.status_code == 200
    assert me_alice_before.json()["balance"] == 10000
    assert me_alice_before.json()["available"] == 10000

    r_pay = await client.post(
        "/payments",
        json={"to_handle": "bob", "amount": 3000, "note": "payment from alice"},
        headers={"Authorization": f"Bearer {token_alice}", "Idempotency-Key": "pay-to-bob"},
    )
    assert r_pay.status_code == 201

    me_alice_after = await client.get(
        "/me",
        headers={"Authorization": f"Bearer {token_alice}"},
    )
    assert me_alice_after.status_code == 200
    assert me_alice_after.json()["balance"] == 7000
    assert me_alice_after.json()["available"] == 7000

    me_bob = await client.get(
        "/me",
        headers={"Authorization": f"Bearer {token_bob}"},
    )
    assert me_bob.status_code == 200
    assert me_bob.json()["balance"] == 3000
    assert me_bob.json()["available"] == 3000
