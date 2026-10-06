"""Dataclasses for stage-2 authorisations.

These shapes are independent of the database layer; the repository
maps between SQLite rows and these dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "AuthorizationRecord",
    "AuthorizationCaptureRecord",
]


_ALLOWED_STATUSES = frozenset({"open", "captured", "voided", "expired"})


@dataclass(frozen=True)
class AuthorizationRecord:
    """An authorisation as stored in the ``authorizations`` table.

    Mirrors the columns of the table. ``captured_amount`` is cumulative
    across all captures of this authorisation; ``latest_payment_id`` is
    the most-recent capture's payment id, or ``None`` if no captures
    have happened yet.
    """

    id: str
    from_user_id: str
    to_user_id: str
    amount: int
    captured_amount: int
    note: str
    visibility: str
    status: str
    expires_at: str
    latest_payment_id: str | None
    created_at: str

    def __post_init__(self) -> None:
        if self.status not in _ALLOWED_STATUSES:
            raise ValueError(
                f"authorization.status must be one of {sorted(_ALLOWED_STATUSES)}, "
                f"got {self.status!r}"
            )
        if self.amount < 1:
            raise ValueError("authorization.amount must be at least 1")
        if self.captured_amount < 0:
            raise ValueError("authorization.captured_amount must be non-negative")
        if self.captured_amount > self.amount:
            raise ValueError("authorization.captured_amount must not exceed amount")


@dataclass(frozen=True)
class AuthorizationCaptureRecord:
    """One capture of an authorisation (extended capture mode support).

    ``seq`` is the per-authorisation 1-based ordinal of this capture;
    the (authorization_id, seq) pair is unique. ``amount`` is the
    amount captured by this specific capture (not the cumulative).
    """

    authorization_id: str
    seq: int
    payment_id: str
    amount: int
    created_at: str

    def __post_init__(self) -> None:
        if self.seq < 1:
            raise ValueError("capture.seq must be >= 1")
        if self.amount < 1:
            raise ValueError("capture.amount must be at least 1")
