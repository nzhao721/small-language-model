"""MUSCIMA++ writer selection for Mashcima handwritten synthesis."""

from __future__ import annotations

import json
import zlib
from collections import OrderedDict
from pathlib import Path

import mashcima as mc
from mashcima.SymbolRepository import SymbolRepository

WRITER_COUNT = 50
DEFAULT_WRITER_MODE = "per-score"
WRITER_MODES = ("per-score", "mixed", "single")
# Keep this small: each SymbolRepository loads a full MUSCIMA++ writer (~100MB+).
_WRITER_REPO_MAX = 2
_VALID_WRITERS_CACHE = (
    Path(__file__).resolve().parents[1] / "data" / "mashcima-cache" / "valid_writers.json"
)

_writer_mode = DEFAULT_WRITER_MODE
_writer_repos: OrderedDict[int, SymbolRepository] = OrderedDict()
_valid_writers: list[int] | None = None


def _get_valid_writers() -> list[int]:
    """Writers that load a complete SymbolRepository (7 of 50 lack tall bar lines)."""
    global _valid_writers
    if _valid_writers is not None:
        return _valid_writers
    if _VALID_WRITERS_CACHE.is_file():
        _valid_writers = json.loads(_VALID_WRITERS_CACHE.read_text(encoding="utf-8"))
        return _valid_writers

    valid: list[int] = []
    for writer in range(1, WRITER_COUNT + 1):
        try:
            SymbolRepository(take_writers=[writer])
            valid.append(writer)
        except AssertionError:
            continue
    if not valid:
        raise RuntimeError("No MUSCIMA++ writers passed SymbolRepository validation")
    _VALID_WRITERS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _VALID_WRITERS_CACHE.write_text(json.dumps(valid), encoding="utf-8")
    _valid_writers = valid
    return _valid_writers


def configure_mashcima_writers(mode: str = DEFAULT_WRITER_MODE) -> None:
    """Configure how MUSCIMA++ writer sprites are chosen for synthesis.

    per-score: one writer per sample key (stable handwriting within each image).
    mixed: all writers' sprites in one pool (random per symbol).
    single: writer 01 only (legacy default).
    """
    global _writer_mode, _valid_writers
    if mode not in WRITER_MODES:
        raise ValueError(f"unknown writer mode {mode!r}; expected one of {WRITER_MODES}")
    _writer_mode = mode
    _writer_repos.clear()
    _valid_writers = None
    if mode == "single":
        mc.use_only_writer_number_one()
    else:
        mc.use_default_setup()


def get_writer_mode() -> str:
    return _writer_mode


def writer_number_for_key(key: str) -> int:
    """Map a sample key to a valid writer 1..50."""
    if _writer_mode == "per-score":
        valid = _get_valid_writers()
        return valid[zlib.crc32(key.encode("utf-8")) % len(valid)]
    return (zlib.crc32(key.encode("utf-8")) % WRITER_COUNT) + 1


def symbol_repository_for_key(key: str) -> SymbolRepository | None:
    """Return a writer-specific repository, or None to use the global default."""
    if _writer_mode != "per-score":
        return None
    writer = writer_number_for_key(key)
    if writer in _writer_repos:
        _writer_repos.move_to_end(writer)
        return _writer_repos[writer]
    while len(_writer_repos) >= _WRITER_REPO_MAX:
        _writer_repos.popitem(last=False)
    _writer_repos[writer] = SymbolRepository(take_writers=[writer])
    return _writer_repos[writer]
