"""
backend/supabase_sync.py — Supabase I/O for the capture/training pipeline.

This is the only Python module that talks to Supabase from the backend
capture flow. server.py and scripts/backfill_supabase.py call into it.

All public functions are fault-tolerant: they log and return a sentinel
rather than raising, except where a caller specifically wants to detect
failure (the backfill CLI exit code).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_BUCKET_IMAGES  = "bottle-images"
_BUCKET_WEIGHTS = "model-weights"

try:
    from supabase import create_client
except ImportError:  # supabase-py not installed — module still imports for tests
    create_client = None  # type: ignore[assignment]


_client_cache: object | None = None
_client_resolved: bool = False


def _log(msg: str) -> None:
    print(f"[Sync] {msg}")


def _logerr(msg: str) -> None:
    print(f"[Sync] {msg}", file=sys.stderr)


def get_client():
    """Return a cached Supabase client, or None if env is not configured."""
    global _client_cache, _client_resolved
    if _client_resolved:
        return _client_cache
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
    if not url or not key or create_client is None:
        _client_cache = None
    else:
        try:
            _client_cache = create_client(url, key)
            _log(f"client connected to {url}")
        except Exception as e:
            _logerr(f"client init failed: {e}")
            _client_cache = None
    _client_resolved = True
    return _client_cache


def parse_class_name(class_name: str) -> tuple[str, str]:
    """
    Split a class name like 'anthony_taylor_advil' into (user, medication).

    Rule: last underscore separates user from medication. The medication may
    contain underscores ('tylenol_500mg') and the user may too ('bob_the_builder').
    A class name with no underscore returns (class_name, '').
    """
    if "_" not in class_name:
        return class_name, ""
    user, _, med = class_name.rpartition("_")
    return user, med
