"""SQLite transaction wrapper using BEGIN IMMEDIATE for write operations.

This module provides a context manager that ensures all write operations
use BEGIN IMMEDIATE to acquire a write lock before any modification,
preventing 'database is locked' errors under concurrent writes.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Callable, TypeVar

T = TypeVar("T")


@contextmanager
def write_transaction(conn: sqlite3.Connection):
    """Context manager for SQLite write transactions with BEGIN IMMEDIATE.

    Usage:
        with write_transaction(conn) as cur:
            cur.execute("UPDATE users SET balance = ? WHERE id = ?", (amount, user_id))

    Or for multiple statements:
        with write_transaction(conn) as cur:
            cur.execute("INSERT INTO payments(...) VALUES(...)", ...)
            cur.execute("INSERT INTO idempotency_keys(...) VALUES(...)", ...)
        # Commits automatically on success
        # Rolls back on any exception

    Args:
        conn: SQLite connection (must have isolation_level=None for autocommit)

    Yields:
        sqlite3.Cursor: A cursor for executing statements within the transaction

    Raises:
        Exception: Any exception during the transaction triggers ROLLBACK
    """
    cursor = conn.cursor()
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield cursor
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def run_in_write_transaction(
    conn: sqlite3.Connection,
    func: Callable[[sqlite3.Cursor], T],
) -> T:
    """Execute a function within a write transaction.

    This is a convenience wrapper for cases where you prefer a function-based
    approach over a context manager.

    Usage:
        def do_writes(cur):
            cur.execute("UPDATE users SET balance = ? WHERE id = ?", (100, "u1"))
            cur.execute("INSERT INTO payments(...) VALUES(...)", ...)
            return result

        result = run_in_write_transaction(conn, do_writes)

    Args:
        conn: SQLite connection
        func: Function that takes a cursor and performs writes

    Returns:
        The return value of func

    Raises:
        Exception: Any exception during the transaction triggers ROLLBACK
    """
    cur = conn.cursor()
    conn.execute("BEGIN IMMEDIATE")
    try:
        result = func(cur)
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
        return result