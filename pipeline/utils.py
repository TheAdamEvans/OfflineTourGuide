"""
Lightweight utilities shared across the scaffolding helpers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib


def iso_timestamp() -> str:
    """
    UTC ISO8601 timestamp suitable for manifests (``...Z`` suffix).
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_path(path: str | Path) -> str:
    """
    Stream a file and return its SHA256 hex digest.
    """
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["iso_timestamp", "sha256_path"]


