import hashlib
import json
from typing import Any, Union

__all__ = ["canonical_body_hash"]


class CanonicalHashError(ValueError):
    """Raised when the input to canonical_body_hash cannot be canonicalised."""


def _canonicalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _canonicalize(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [_canonicalize(item) for item in obj]
    return obj


def canonical_body_hash(body: Union[bytes, bytearray, str, dict, list, None]) -> str:
    """Return a deterministic SHA-256 hex digest for the canonical form of ``body``.

    The canonical form sorts dictionary keys recursively and uses the
    ``(",", ":")`` separators with ``sort_keys=True``. Whitespace and key
    order in the input therefore do not affect the resulting hash. Two
    payloads that are equal as JSON values (after parsing) always hash
    to the same digest.

    Accepts:

    * ``bytes``/``bytearray`` — raw JSON request body
    * ``str`` — raw JSON request body
    * ``dict``/``list`` — already-parsed JSON value (typically the body the
      application has just deserialised)

    Any other type, or a string/bytes payload that does not parse as
    JSON, raises :class:`CanonicalHashError`.
    """
    if isinstance(body, (bytes, bytearray)):
        try:
            text = bytes(body).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CanonicalHashError("body is not valid UTF-8 JSON") from exc
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CanonicalHashError(
                f"body is not valid JSON: {exc.msg}"
            ) from exc
    elif isinstance(body, str):
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise CanonicalHashError(
                f"body is not valid JSON: {exc.msg}"
            ) from exc
    elif isinstance(body, (dict, list)):
        parsed = body
    elif body is None:
        raise CanonicalHashError("body is None, expected JSON value")
    else:
        raise CanonicalHashError(
            f"unsupported body type: {type(body).__name__}"
        )

    if not isinstance(parsed, (dict, list)):
        raise CanonicalHashError(
            "body must be a JSON object or array at the top level"
        )

    canonical = _canonicalize(parsed)
    serialized = json.dumps(
        canonical, separators=(",", ":"), sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
