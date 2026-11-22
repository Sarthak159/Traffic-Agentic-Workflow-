"""Canonical serialization and content hashing shared by every artifact identity."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible value deterministically (sorted keys, no whitespace, no NaN)."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def sha256_hex(data: bytes | str) -> str:
    """Return the hex SHA-256 digest of bytes or UTF-8 text."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def fingerprint(value: Any) -> str:
    """Return the SHA-256 of the canonical JSON form of a value."""
    return sha256_hex(canonical_json(value))


def file_sha256(path: Any, chunk_size: int = 1 << 20) -> str:
    """Hash a file's bytes without loading it into memory at once."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
