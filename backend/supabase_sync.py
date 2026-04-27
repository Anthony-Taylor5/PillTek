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


_RETRY_DELAY_S = 2.0


def _iter_dataset_files(dataset_dir: Path):
    """Yield (split, file_path) for every jpg under images/{train,val}/."""
    for split in ("train", "val"):
        d = dataset_dir / "images" / split
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.jpg")):
            yield split, p


def _upload_one_image(client, class_name: str, split: str, fp: Path,
                      user_id: str, med: str) -> bool:
    """Upload a single jpg + insert its row. Returns True on success."""
    storage_path = f"{class_name}/{split}/{fp.name}"
    try:
        with fp.open("rb") as fh:
            client.storage.from_(_BUCKET_IMAGES).upload(
                storage_path,
                fh.read(),
                {"content-type": "image/jpeg", "upsert": "true"},
            )
        client.table("bottle_images").insert({
            "class_name":      class_name,
            "user_identifier": user_id,
            "medication_name": med,
            "split":           split,
            "filename":        fp.name,
            "storage_path":    storage_path,
        }).execute()
        return True
    except Exception as e:
        _logerr(f"image upload failed for {fp.name}: {e}")
        return False


def upload_dataset_images(class_name: str, dataset_dir: Path) -> tuple[int, int]:
    """
    Upload every jpg under <dataset_dir>/images/{train,val}/ to Supabase
    Storage and insert one bottle_images row per file.

    Returns (uploaded_count, failed_count). One failure does not abort the
    batch. Failed files are retried once after _RETRY_DELAY_S seconds.
    """
    client = get_client()
    if client is None:
        _log(f"skipping image upload for {class_name} (no Supabase client)")
        return (0, 0)

    user_id, med = parse_class_name(class_name)
    files = list(_iter_dataset_files(Path(dataset_dir)))
    _log(f"uploading {len(files)} images for {class_name}…")
    if not files:
        return (0, 0)

    t0 = time.monotonic()
    failures: list[tuple[str, Path]] = []
    uploaded = 0
    for split, fp in files:
        if _upload_one_image(client, class_name, split, fp, user_id, med):
            uploaded += 1
        else:
            failures.append((split, fp))

    if failures:
        _log(f"retrying {len(failures)} failed uploads in {_RETRY_DELAY_S}s…")
        time.sleep(_RETRY_DELAY_S)
        still_failing: list[tuple[str, Path]] = []
        for split, fp in failures:
            if _upload_one_image(client, class_name, split, fp, user_id, med):
                uploaded += 1
            else:
                still_failing.append((split, fp))
        failures = still_failing

    failed = len(failures)
    dt = time.monotonic() - t0
    _log(f"✓ uploaded {uploaded}/{len(files)} ({failed} failed) in {dt:.1f}s")
    return (uploaded, failed)
