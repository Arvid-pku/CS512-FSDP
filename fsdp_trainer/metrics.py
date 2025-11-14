"""Lightweight JSONL metrics logger for distributed experiments."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import ensure_dir, is_rank_zero


class MetricsLogger:
    """Writes structured metrics events to a JSONL file on rank zero only."""

    def __init__(self, path: Optional[str]) -> None:
        self._path = Path(path) if path else None
        self._fh = None
        if self._path is not None and is_rank_zero():
            ensure_dir(self._path.parent)
            self._fh = self._path.open("a", encoding="utf-8")

    def log(self, payload: Dict[str, Any]) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(payload) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

