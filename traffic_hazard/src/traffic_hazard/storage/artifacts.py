"""Private run artifacts: atomic JSON writes, JSONL appends, and output-directory safety."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class OutputExistsError(FileExistsError):
    pass


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    """Create `path`; refuse a non-empty directory unless `overwrite` (completed runs are never
    clobbered).
    """
    path = Path(path)
    if path.exists() and not path.is_dir():
        raise OutputExistsError(f"output path {path} exists and is not a directory")
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise OutputExistsError(f"output directory {path} is not empty (use --overwrite)")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, value: Any) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    write_bytes_atomic(path, text.encode("utf-8"))


class JsonlWriter:
    """Append-only JSONL file; each record is one line, flushed as written."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        self.count = 0

    def write(self, record: Any) -> None:
        self._handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self._handle.flush()
        self.count += 1

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
