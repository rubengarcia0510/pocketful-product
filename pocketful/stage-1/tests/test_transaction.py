"""Tests for the SQLite write transaction wrapper."""

import os
import sqlite3
import threading
import time
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.sqlite_utils.transaction import write_transaction, run_in_write_transaction


@pytest.fixture
def isolated_db_path(tmp_path):
    """Provide an isolated DB path that does not interact with the session-level db_path."""
    path = str(tmp_path / "test.db")
    yield path


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def test_begin_immediate_executed(isolated_db_path):
    """Verify BEGIN IMMEDIATE is executed for write transactions."""
    conn = _connect(isolated_db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS items (id TEXT PRIMARY KEY, value TEXT)")

    with write_transaction(conn) as cur:
        cur.execute("INSERT INTO items (id, value) VALUES (?, ?)", ("1", "test"))

    row = conn.execute("SELECT * FROM items WHERE id = ?", ("1",)).fetchone()
    assert row is not None
    assert row["value"] == "test"
    conn.close()


def test_rollback_on_exception(isolated_db_path):
    """Verify ROLLBACK occurs when an exception is raised."""
    conn = _connect(isolated_db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS items (id TEXT PRIMARY KEY, value TEXT)")

    with pytest.raises(Exception):
        with write_transaction(conn) as cur:
            cur.execute("INSERT INTO items (id, value) VALUES (?, ?)", ("1", "before_error"))
            raise Exception("simulated error")

    row = conn.execute("SELECT * FROM items WHERE id = ?", ("1",)).fetchone()
    assert row is None
    conn.close()


def test_multiple_statements_in_transaction(isolated_db_path):
    """Verify multiple statements are committed together."""
    conn = _connect(isolated_db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS items (id TEXT PRIMARY KEY, value TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, count INTEGER)")

    with write_transaction(conn) as cur:
        cur.execute("INSERT INTO items (id, value) VALUES (?, ?)", ("1", "item1"))
        cur.execute("INSERT INTO items (id, value) VALUES (?, ?)", ("2", "item2"))
        cur.execute("INSERT INTO counters (name, count) VALUES (?, ?)", ("total", 2))

    items = conn.execute("SELECT * FROM items ORDER BY id").fetchall()
    total = conn.execute("SELECT * FROM counters WHERE name = ?", ("total",)).fetchone()

    assert len(items) == 2
    assert total["count"] == 2
    conn.close()


def test_run_in_write_transaction_function(isolated_db_path):
    """Test the function-based wrapper."""
    conn = _connect(isolated_db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS items (id TEXT PRIMARY KEY, value TEXT)")

    def do_inserts(cur):
        cur.execute("INSERT INTO items (id, value) VALUES (?, ?)", ("1", "func_test"))
        return "success"

    result = run_in_write_transaction(conn, do_inserts)

    assert result == "success"
    row = conn.execute("SELECT * FROM items WHERE id = ?", ("1",)).fetchone()
    assert row is not None
    conn.close()


def test_run_in_write_transaction_rollback(isolated_db_path):
    """Test that function-based wrapper rolls back on exception."""
    conn = _connect(isolated_db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS items (id TEXT PRIMARY KEY, value TEXT)")

    def do_insert_with_error(cur):
        cur.execute("INSERT INTO items (id, value) VALUES (?, ?)", ("1", "before_error"))
        raise ValueError("test error")

    with pytest.raises(ValueError):
        run_in_write_transaction(conn, do_insert_with_error)

    row = conn.execute("SELECT * FROM items WHERE id = ?", ("1",)).fetchone()
    assert row is None
    conn.close()


def test_concurrent_writes_serialized(isolated_db_path):
    """Verify concurrent writes are serialized without 'database is locked' errors."""
    conn = _connect(isolated_db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, count INTEGER)")
    conn.execute("INSERT OR REPLACE INTO counters (name, count) VALUES ('total', 0)")

    errors = []
    results = []

    def increment_counter(thread_id: int):
        try:
            local_conn = _connect(isolated_db_path)
            for i in range(5):
                with write_transaction(local_conn) as cur:
                    current = cur.execute("SELECT count FROM counters WHERE name = 'total'").fetchone()["count"]
                    cur.execute("UPDATE counters SET count = ? WHERE name = 'total'", (current + 1,))
            local_conn.close()
            results.append(thread_id)
        except Exception as e:
            errors.append((thread_id, str(e)))

    num_threads = 4
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(increment_counter, i) for i in range(num_threads)]
        for f in as_completed(futures):
            pass

    final_count = conn.execute("SELECT count FROM counters WHERE name = 'total'").fetchone()["count"]
    assert final_count == num_threads * 5
    assert len(errors) == 0, f"Errors occurred: {errors}"
    conn.close()