# Bottle Image + Model Supabase Sync — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After bottle capture completes, sync all images to Supabase, fine-tune the YOLO model in a per-class run folder, then upload the trained weights and record metadata in a new `user_models` table — and provide a one-shot backfill for existing local data.

**Architecture:** Add a single `backend/supabase_sync.py` module that owns every Supabase call from the Python pipeline. Wire it into `backend/server.py`'s `/capture-frame` (image upload) and `_trigger_training` (model row + weights upload). Replace the `runs/user_tuned_<N>/` naming with `runs/<class_name>[_v<N>]/`. Provide `scripts/backfill_supabase.py` for existing data.

**Tech Stack:** Python 3.11+, Flask, supabase-py 2.x, ultralytics (YOLO), pytest + unittest.mock.

**Spec:** `docs/superpowers/specs/2026-04-26-bottle-image-and-model-supabase-sync-design.md`

---

## File Map

- **Create** `backend/supabase_sync.py` — all Supabase I/O for the capture pipeline.
- **Create** `scripts/backfill_supabase.py` — one-shot CLI for existing local data.
- **Create** `tests/test_supabase_sync.py` — pure-helper tests + mocked-client tests.
- **Create** `tests/test_class_run_dir.py` — pure helper for run-folder naming.
- **Modify** `supabase/schema.sql` — extend `bottle_images`, add `user_models`, add `model-weights` bucket.
- **Modify** `capture_bottles.py` — replace `next_run_dir(prefix=...)` call with `class_run_dir(...)`; add `--run-name` CLI override.
- **Modify** `backend/server.py` — fire image upload thread on capture-done; wrap `_trigger_training` to insert/update `user_models` and upload weights.

---

## Task 1: Schema migration

**Files:**
- Modify: `supabase/schema.sql` (append at end of file)

- [ ] **Step 1: Append migration block to schema**

Append exactly this block to the bottom of `supabase/schema.sql`:

```sql
-- ── 2026-04-26: bottle_images extension + user_models + model-weights bucket ──

-- Extend bottle_images (backward-compatible). Existing mobile-app rows that set
-- medication_id keep working; backend pipeline rows store strings instead.
ALTER TABLE bottle_images ALTER COLUMN medication_id DROP NOT NULL;
ALTER TABLE bottle_images ADD COLUMN IF NOT EXISTS user_identifier text;
ALTER TABLE bottle_images ADD COLUMN IF NOT EXISTS medication_name text;
ALTER TABLE bottle_images ADD COLUMN IF NOT EXISTS class_name      text;
ALTER TABLE bottle_images ADD COLUMN IF NOT EXISTS split           text
  CHECK (split IN ('train','val'));

-- Trained per-user model registry.
CREATE TABLE IF NOT EXISTS user_models (
  id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_identifier      text NOT NULL,
  medication_name      text NOT NULL,
  model_name           text NOT NULL,
  base_model           text,
  dataset_path         text,
  weights_local_path   text,
  weights_storage_path text,
  version              integer DEFAULT 1,
  status               text DEFAULT 'ready'
                            CHECK (status IN ('training','ready','failed')),
  created_at           timestamptz DEFAULT now()
);
ALTER TABLE user_models DISABLE ROW LEVEL SECURITY;

-- Storage bucket for trained .pt weights.
-- May fail under non-service-role connections — fallback is creating it manually
-- in Dashboard → Storage → New bucket. Name: model-weights, Public: false.
INSERT INTO storage.buckets (id, name, public)
  VALUES ('model-weights', 'model-weights', false)
  ON CONFLICT DO NOTHING;
```

- [ ] **Step 2: Commit**

```bash
git add supabase/schema.sql
git commit -m "Add bottle_images extension, user_models table, model-weights bucket"
```

The user will run this migration manually in the Supabase SQL editor (Dashboard → SQL Editor). That happens in Task 12.

---

## Task 2: `class_run_dir` helper (TDD)

A pure helper that picks the next run folder name based on what exists on disk.

**Files:**
- Create: `tests/test_class_run_dir.py`
- Modify: `capture_bottles.py` (add `class_run_dir` next to existing `next_run_dir`)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_class_run_dir.py`:

```python
"""Tests for capture_bottles.class_run_dir()."""
from pathlib import Path

from capture_bottles import class_run_dir


def test_first_run_uses_bare_class_name(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    assert class_run_dir(runs, "anthony_taylor_advil") == runs / "anthony_taylor_advil"


def test_second_run_appends_v2(tmp_path):
    runs = tmp_path / "runs"
    (runs / "anthony_taylor_advil").mkdir(parents=True)
    assert class_run_dir(runs, "anthony_taylor_advil") == runs / "anthony_taylor_advil_v2"


def test_skips_to_next_available_version(tmp_path):
    runs = tmp_path / "runs"
    (runs / "anthony_taylor_advil").mkdir(parents=True)
    (runs / "anthony_taylor_advil_v2").mkdir()
    (runs / "anthony_taylor_advil_v3").mkdir()
    assert class_run_dir(runs, "anthony_taylor_advil") == runs / "anthony_taylor_advil_v4"


def test_unrelated_class_unaffected(tmp_path):
    runs = tmp_path / "runs"
    (runs / "anthony_taylor_advil").mkdir(parents=True)
    assert class_run_dir(runs, "bob_tylenol") == runs / "bob_tylenol"
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_class_run_dir.py -v
```

Expected: 4 errors with `ImportError: cannot import name 'class_run_dir'`.

- [ ] **Step 3: Implement `class_run_dir`**

In `capture_bottles.py`, immediately after the existing `next_run_dir` function (around line 236), add:

```python
def class_run_dir(runs_dir: Path, class_name: str) -> Path:
    """
    Choose a run folder named after the class (e.g. anthony_taylor_advil).
    On retrain, append _v2, _v3, ... so older runs aren't overwritten.
    """
    base = runs_dir / class_name
    if not base.exists():
        return base
    i = 2
    while (runs_dir / f"{class_name}_v{i}").exists():
        i += 1
    return runs_dir / f"{class_name}_v{i}"
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_class_run_dir.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add capture_bottles.py tests/test_class_run_dir.py
git commit -m "Add class_run_dir helper for per-class run folder naming"
```

---

## Task 3: Wire `class_run_dir` into `capture_bottles.py` main path + add `--run-name` override

The server-side wrapper needs to pin the exact run folder so the local path matches the `user_models.version` it just inserted. Add `--run-name` for that, otherwise fall back to `class_run_dir`.

**Files:**
- Modify: `capture_bottles.py:702-708` (CLI argparse — add `--run-name`)
- Modify: `capture_bottles.py:754` and `capture_bottles.py:808` (replace both `next_run_dir(...)` call sites)

- [ ] **Step 1: Add `--run-name` arg**

In `capture_bottles.py`, inside `main()` next to the other parser.add_argument calls (around line 708, after `--runs-dir`), add:

```python
    parser.add_argument(
        "--run-name",
        default=None,
        help="Exact run folder name under --runs-dir. If omitted, derived from "
             "--class-name via class_run_dir() (anthony_taylor_advil, _v2, _v3, ...).",
    )
```

- [ ] **Step 2: Replace both `next_run_dir` call sites**

There are two call sites that need updating.

In the **train-only branch**, replace:

```python
        run_dir = next_run_dir(runs_dir, prefix="user_tuned_")
```

with:

```python
        run_dir = (runs_dir / args.run_name) if args.run_name else class_run_dir(runs_dir, args.class_name)
        run_dir.mkdir(parents=True, exist_ok=True)
```

In the **full capture+train branch**, replace:

```python
    run_dir = next_run_dir(runs_dir, prefix="user_tuned_")
```

with the same two-line block.

- [ ] **Step 3: Smoke-check the CLI parses**

```bash
python capture_bottles.py --help | grep -E "run-name|class-name"
```

Expected: both `--run-name` and `--class-name` appear in the output.

- [ ] **Step 4: Commit**

```bash
git add capture_bottles.py
git commit -m "Use class_run_dir for run folder naming; add --run-name override"
```

---

## Task 4: `parse_class_name` helper + module skeleton (TDD)

Pure helper to split `<user>_<medication>` on the **last** underscore.

**Files:**
- Create: `backend/supabase_sync.py`
- Create: `tests/test_supabase_sync.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_supabase_sync.py`:

```python
"""Tests for backend/supabase_sync.py."""
import sys
from pathlib import Path

# Ensure backend/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.supabase_sync import parse_class_name


def test_parse_two_token_class_name():
    assert parse_class_name("anthony_taylor_advil") == ("anthony_taylor", "advil")


def test_parse_multi_token_medication():
    # Last underscore separates user from medication; medication may contain '_'.
    # The split rule is: user = everything before the last '_'.
    assert parse_class_name("bob_the_builder_advil") == ("bob_the_builder", "advil")


def test_parse_single_token_falls_back():
    # No underscore → user = full string, medication = ''.
    assert parse_class_name("solo") == ("solo", "")


def test_parse_empty_string():
    assert parse_class_name("") == ("", "")
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_supabase_sync.py -v
```

Expected: 4 errors with `ModuleNotFoundError: No module named 'backend.supabase_sync'`.

- [ ] **Step 3: Create the module skeleton with `parse_class_name`**

Create `backend/supabase_sync.py`:

```python
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
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_supabase_sync.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/supabase_sync.py tests/test_supabase_sync.py
git commit -m "Add supabase_sync module skeleton with parse_class_name"
```

---

## Task 5: Cached Supabase client `get_client()` (TDD)

**Files:**
- Modify: `backend/supabase_sync.py`
- Modify: `tests/test_supabase_sync.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_supabase_sync.py`:

```python
import importlib

from unittest.mock import patch


def _reload_module():
    import backend.supabase_sync as m
    importlib.reload(m)
    return m


def test_get_client_returns_none_when_env_missing(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    m = _reload_module()
    assert m.get_client() is None


def test_get_client_returns_cached_client_when_env_present(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service_key_xxx")
    m = _reload_module()
    fake_client = object()
    with patch("backend.supabase_sync.create_client", return_value=fake_client) as cc:
        assert m.get_client() is fake_client
        assert m.get_client() is fake_client  # cached — second call doesn't recreate
        assert cc.call_count == 1
```

- [ ] **Step 2: Run tests — verify the new ones fail**

```bash
pytest tests/test_supabase_sync.py::test_get_client_returns_none_when_env_missing tests/test_supabase_sync.py::test_get_client_returns_cached_client_when_env_present -v
```

Expected: 2 failures (`AttributeError: module ... has no attribute 'get_client'`).

- [ ] **Step 3: Implement `get_client`**

Add to `backend/supabase_sync.py`, near the top (after the bucket constants):

```python
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
```

- [ ] **Step 4: Run all tests in the file — verify pass**

```bash
pytest tests/test_supabase_sync.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/supabase_sync.py tests/test_supabase_sync.py
git commit -m "Add cached Supabase client accessor in supabase_sync"
```

---

## Task 6: `upload_dataset_images` (TDD with mock client)

Walk a dataset directory, upload every JPEG to storage, insert one `bottle_images` row per file.

**Files:**
- Modify: `backend/supabase_sync.py`
- Modify: `tests/test_supabase_sync.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_supabase_sync.py`:

```python
def _make_dataset(tmp_path: Path, n_train: int = 2, n_val: int = 1) -> Path:
    base = tmp_path / "anthony_taylor_advil"
    (base / "images" / "train").mkdir(parents=True)
    (base / "images" / "val").mkdir(parents=True)
    for i in range(n_train):
        (base / "images" / "train" / f"bottle_{i:03d}.jpg").write_bytes(b"\xff\xd8fake")
    for i in range(n_val):
        (base / "images" / "val" / f"bottle_v{i:03d}.jpg").write_bytes(b"\xff\xd8fakeV")
    return base


def _make_mock_client():
    """Mock Supabase client where storage.upload + table.insert both succeed."""
    from unittest.mock import MagicMock
    client = MagicMock()
    client.storage.from_.return_value.upload.return_value = None
    client.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[{"id": "row-id"}])
    return client


def test_upload_dataset_images_uploads_every_file(tmp_path, monkeypatch):
    base = _make_dataset(tmp_path, n_train=2, n_val=1)
    m = _reload_module()
    client = _make_mock_client()
    monkeypatch.setattr(m, "get_client", lambda: client)

    uploaded, failed = m.upload_dataset_images("anthony_taylor_advil", base)
    assert (uploaded, failed) == (3, 0)

    # Each file → one storage upload + one DB insert
    assert client.storage.from_.return_value.upload.call_count == 3
    assert client.table.return_value.insert.call_count == 3


def test_upload_dataset_images_records_split_and_strings(tmp_path, monkeypatch):
    base = _make_dataset(tmp_path, n_train=1, n_val=1)
    m = _reload_module()
    client = _make_mock_client()
    monkeypatch.setattr(m, "get_client", lambda: client)

    m.upload_dataset_images("anthony_taylor_advil", base)

    inserted_rows = [
        call.args[0]
        for call in client.table.return_value.insert.call_args_list
    ]
    splits = sorted(r["split"] for r in inserted_rows)
    assert splits == ["train", "val"]
    for r in inserted_rows:
        assert r["user_identifier"] == "anthony_taylor"
        assert r["medication_name"] == "advil"
        assert r["class_name"]      == "anthony_taylor_advil"
        assert r["filename"].startswith("bottle_")
        assert r["storage_path"].startswith("anthony_taylor_advil/")


def test_upload_dataset_images_continues_on_per_file_failure(tmp_path, monkeypatch):
    base = _make_dataset(tmp_path, n_train=2, n_val=1)
    m = _reload_module()
    client = _make_mock_client()

    # Make the SECOND upload raise; first and third succeed.
    seq = [None, RuntimeError("network blip"), None, None, None, None]  # extra slots for retry
    client.storage.from_.return_value.upload.side_effect = seq
    monkeypatch.setattr(m, "get_client", lambda: client)
    monkeypatch.setattr(m, "_RETRY_DELAY_S", 0.0)  # don't actually sleep in tests

    uploaded, failed = m.upload_dataset_images("anthony_taylor_advil", base)
    # Retry succeeds on the second pass → ends up 3/0.
    assert uploaded == 3 and failed == 0


def test_upload_dataset_images_returns_zero_when_no_client(tmp_path, monkeypatch):
    base = _make_dataset(tmp_path)
    m = _reload_module()
    monkeypatch.setattr(m, "get_client", lambda: None)

    uploaded, failed = m.upload_dataset_images("anthony_taylor_advil", base)
    assert (uploaded, failed) == (0, 0)
```

- [ ] **Step 2: Run new tests — verify they fail**

```bash
pytest tests/test_supabase_sync.py -v -k upload_dataset_images
```

Expected: 4 failures (`AttributeError: ... has no attribute 'upload_dataset_images'`).

- [ ] **Step 3: Implement `upload_dataset_images`**

Append to `backend/supabase_sync.py`:

```python
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
```

- [ ] **Step 4: Run all `supabase_sync` tests — verify pass**

```bash
pytest tests/test_supabase_sync.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/supabase_sync.py tests/test_supabase_sync.py
git commit -m "Add upload_dataset_images with per-file retry"
```

---

## Task 7: Model row insert/update + version computation (TDD)

**Files:**
- Modify: `backend/supabase_sync.py`
- Modify: `tests/test_supabase_sync.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_supabase_sync.py`:

```python
def _mock_client_with_models(rows):
    """Mock client whose user_models query returns the given existing rows."""
    from unittest.mock import MagicMock
    client = MagicMock()
    client.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(data=rows)
    client.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[{"id": "new-row"}])
    client.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock(data=[{}])
    return client


def test_compute_next_version_no_existing(monkeypatch):
    m = _reload_module()
    client = _mock_client_with_models([])
    monkeypatch.setattr(m, "get_client", lambda: client)
    assert m.compute_next_version("anthony_taylor_advil") == 1


def test_compute_next_version_with_existing(monkeypatch):
    m = _reload_module()
    client = _mock_client_with_models([
        {"version": 1}, {"version": 2}, {"version": 3},
    ])
    monkeypatch.setattr(m, "get_client", lambda: client)
    assert m.compute_next_version("anthony_taylor_advil") == 4


def test_insert_model_row_returns_id(monkeypatch):
    m = _reload_module()
    client = _mock_client_with_models([])
    monkeypatch.setattr(m, "get_client", lambda: client)
    rid = m.insert_model_row(
        class_name="anthony_taylor_advil",
        base_model="v10",
        dataset_path="user_bottles/anthony_taylor_advil",
        version=1,
        status="training",
    )
    assert rid == "new-row"
    inserted = client.table.return_value.insert.call_args.args[0]
    assert inserted["model_name"]      == "anthony_taylor_advil"
    assert inserted["user_identifier"] == "anthony_taylor"
    assert inserted["medication_name"] == "advil"
    assert inserted["base_model"]      == "v10"
    assert inserted["version"]         == 1
    assert inserted["status"]          == "training"


def test_update_model_status_passes_fields(monkeypatch):
    m = _reload_module()
    client = _mock_client_with_models([])
    monkeypatch.setattr(m, "get_client", lambda: client)
    m.update_model_status("model-id-123", status="ready",
                          weights_local_path="runs/foo/weights/best.pt",
                          weights_storage_path="model-weights/foo/v1/best.pt")
    payload = client.table.return_value.update.call_args.args[0]
    assert payload["status"]               == "ready"
    assert payload["weights_local_path"]   == "runs/foo/weights/best.pt"
    assert payload["weights_storage_path"] == "model-weights/foo/v1/best.pt"
```

- [ ] **Step 2: Run new tests — verify they fail**

```bash
pytest tests/test_supabase_sync.py -v -k "compute_next_version or insert_model_row or update_model_status"
```

Expected: 4 failures (missing attributes).

- [ ] **Step 3: Implement the three helpers**

Append to `backend/supabase_sync.py`:

```python
def compute_next_version(class_name: str) -> int:
    """Return max(version)+1 across existing user_models rows for class_name, or 1."""
    client = get_client()
    if client is None:
        return 1
    try:
        res = client.table("user_models").select("version").eq("model_name", class_name).execute()
        rows = res.data or []
        if not rows:
            return 1
        return max(int(r.get("version") or 0) for r in rows) + 1
    except Exception as e:
        _logerr(f"compute_next_version failed: {e}")
        return 1


def insert_model_row(class_name: str, base_model: str, dataset_path: str,
                     version: int, status: str = "training") -> str | None:
    """Insert a user_models row. Returns the new row's id, or None on failure."""
    client = get_client()
    if client is None:
        return None
    user_id, med = parse_class_name(class_name)
    payload = {
        "user_identifier": user_id,
        "medication_name": med,
        "model_name":      class_name,
        "base_model":      base_model,
        "dataset_path":    dataset_path,
        "version":         version,
        "status":          status,
    }
    try:
        res = client.table("user_models").insert(payload).execute()
        rid = (res.data or [{}])[0].get("id")
        _log(f"inserted user_models row id={rid} status={status} v{version}")
        return rid
    except Exception as e:
        _logerr(f"insert_model_row failed: {e}")
        return None


def update_model_status(model_id: str, status: str, **fields) -> None:
    """Update an existing user_models row. Never raises."""
    client = get_client()
    if client is None or not model_id:
        return
    payload = {"status": status, **fields}
    try:
        client.table("user_models").update(payload).eq("id", model_id).execute()
        _log(f"user_models id={model_id} → status={status}")
    except Exception as e:
        _logerr(f"update_model_status failed: {e}")
```

- [ ] **Step 4: Run all tests — verify pass**

```bash
pytest tests/test_supabase_sync.py -v
```

Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/supabase_sync.py tests/test_supabase_sync.py
git commit -m "Add user_models insert/update + version computation"
```

---

## Task 8: `upload_model_weights` (TDD)

Upload `best.pt` to the `model-weights` bucket.

**Files:**
- Modify: `backend/supabase_sync.py`
- Modify: `tests/test_supabase_sync.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_supabase_sync.py`:

```python
def test_upload_model_weights_uploads_file(tmp_path, monkeypatch):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"\x80\x02fake-weights")
    m = _reload_module()
    client = _mock_client_with_models([])
    monkeypatch.setattr(m, "get_client", lambda: client)

    storage_path = m.upload_model_weights(
        class_name="anthony_taylor_advil",
        weights_path=weights,
        version=1,
    )
    assert storage_path == "anthony_taylor_advil/v1/best.pt"
    upload_call = client.storage.from_.return_value.upload.call_args
    assert upload_call.args[0] == "anthony_taylor_advil/v1/best.pt"


def test_upload_model_weights_returns_none_when_missing_file(tmp_path, monkeypatch):
    m = _reload_module()
    client = _mock_client_with_models([])
    monkeypatch.setattr(m, "get_client", lambda: client)
    assert m.upload_model_weights(
        class_name="anthony_taylor_advil",
        weights_path=tmp_path / "nope.pt",
        version=1,
    ) is None


def test_upload_model_weights_returns_none_without_client(tmp_path, monkeypatch):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    m = _reload_module()
    monkeypatch.setattr(m, "get_client", lambda: None)
    assert m.upload_model_weights(
        class_name="anthony_taylor_advil",
        weights_path=weights,
        version=2,
    ) is None
```

- [ ] **Step 2: Run new tests — verify they fail**

```bash
pytest tests/test_supabase_sync.py -v -k upload_model_weights
```

Expected: 3 failures (`AttributeError: ... has no attribute 'upload_model_weights'`).

- [ ] **Step 3: Implement**

Append to `backend/supabase_sync.py`:

```python
def upload_model_weights(class_name: str, weights_path: Path,
                         version: int) -> str | None:
    """
    Upload best.pt to model-weights/<class_name>/v<version>/best.pt.
    Returns the storage path on success, None on failure.
    """
    client = get_client()
    if client is None:
        return None
    weights_path = Path(weights_path)
    if not weights_path.exists():
        _logerr(f"weights file not found: {weights_path}")
        return None
    storage_path = f"{class_name}/v{version}/{weights_path.name}"
    size_mb = weights_path.stat().st_size / 1024 / 1024
    _log(f"uploading weights → model-weights/{storage_path} ({size_mb:.1f} MB)")
    try:
        with weights_path.open("rb") as fh:
            client.storage.from_(_BUCKET_WEIGHTS).upload(
                storage_path,
                fh.read(),
                {"content-type": "application/octet-stream", "upsert": "true"},
            )
        _log(f"✓ weights uploaded ({size_mb:.1f} MB)")
        return storage_path
    except Exception as e:
        _logerr(f"weights upload failed: {e}")
        return None
```

- [ ] **Step 4: Run all tests — verify pass**

```bash
pytest tests/test_supabase_sync.py -v
```

Expected: 17 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/supabase_sync.py tests/test_supabase_sync.py
git commit -m "Add upload_model_weights to model-weights bucket"
```

---

## Task 9: Wire image-upload thread into `/capture-frame` (final-frame trigger)

When the last frame is captured (status flips to `done`), spawn a background thread that uploads the dataset to Supabase. Training continues to run independently in its own thread.

**Files:**
- Modify: `backend/server.py:621-622` (after `if done >= session['total']: _trigger_training(session)`)
- Modify: `backend/server.py` import block (add supabase_sync import)

- [ ] **Step 1: Add the import**

Near the top of `backend/server.py`, after the existing `from supabase import create_client` block (around line 47), add:

```python
from backend import supabase_sync
```

If that import path fails because the file is run as `python backend/server.py` (not as a package), use this fallback form instead — both work:

```python
try:
    from backend import supabase_sync
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    import supabase_sync  # type: ignore[no-redef]
```

- [ ] **Step 2: Spawn the upload thread on capture-done**

In `capture_frame()`, locate the block at line 619-622:

```python
    print(f"[Capture:{session_id[:8]}] {done}/{session['total']} saved (box={box_type})")

    if done >= session['total']:
        _trigger_training(session)
```

Replace it with:

```python
    print(f"[Capture:{session_id[:8]}] {done}/{session['total']} saved (box={box_type})")

    if done >= session['total']:
        _trigger_image_upload(session)
        _trigger_training(session)
```

- [ ] **Step 3: Add the helper just above `_trigger_training`**

In `backend/server.py`, immediately above the existing `def _trigger_training(session):` (around line 307), insert:

```python
def _trigger_image_upload(session):
    """Spawn dataset-image upload in a background thread. Non-blocking."""
    def _run():
        sid = session['session_id'][:8]
        class_name  = session['class_name']
        dataset_dir = Path(_REPO_ROOT) / 'user_bottles' / class_name
        print(f"[Sync:{sid}] starting dataset image upload for {class_name}")
        try:
            uploaded, failed = supabase_sync.upload_dataset_images(class_name, dataset_dir)
            print(f"[Sync:{sid}] dataset upload finished: {uploaded} uploaded, {failed} failed")
        except Exception as e:
            print(f"[Sync:{sid}] dataset upload errored: {e}", file=sys.stderr)
    threading.Thread(target=_run, daemon=True).start()
```

You'll also need `from pathlib import Path` at the top of the file (check whether it's already imported — it isn't in the current `server.py`, so add it).

- [ ] **Step 4: Smoke-check the file imports**

```bash
python -c "import backend.server"
```

Expected: no traceback. (May print Supabase init lines.)

- [ ] **Step 5: Commit**

```bash
git add backend/server.py
git commit -m "Trigger Supabase image upload after final capture frame"
```

---

## Task 10: Wrap `_trigger_training` to insert/update `user_models` and upload weights

Replace the existing `_trigger_training` body so it: (1) inserts a row, (2) runs the existing subprocess, pinning the run name, (3) uploads weights and updates the row on success / marks failed otherwise.

**Files:**
- Modify: `backend/server.py:307-339` (the `_trigger_training` function)

- [ ] **Step 1: Replace `_trigger_training`**

Replace the entire existing `_trigger_training` function (lines 307-339) with:

```python
def _trigger_training(session):
    """Insert user_models row, fine-tune via subprocess, upload weights, finalize row."""
    def _train():
        sid        = session['session_id'][:8]
        class_name = session['class_name']
        version    = supabase_sync.compute_next_version(class_name)
        run_name   = class_name if version == 1 else f"{class_name}_v{version}"
        run_dir    = os.path.join(_REPO_ROOT, 'runs', run_name)
        weights_local = os.path.join(run_dir, 'weights', 'best.pt')

        # Pick a friendly base_model label: 'v10' if YOLO_WEIGHTS lives under
        # runs/.../train_v10/..., else the full path.
        base_label = 'v10' if 'train_v10' in _YOLO_WEIGHTS else _YOLO_WEIGHTS

        model_id = supabase_sync.insert_model_row(
            class_name=class_name,
            base_model=base_label,
            dataset_path=os.path.join('user_bottles', class_name),
            version=version,
            status='training',
        )

        cmd = [
            sys.executable, os.path.join(_REPO_ROOT, 'capture_bottles.py'),
            '--class-name',   class_name,
            '--weights',      _YOLO_WEIGHTS,
            '--dataset-dir',  'user_bottles',
            '--runs-dir',     'runs',
            '--run-name',     run_name,
            '--train-only',
        ]
        print(f"[Training:{sid}] starting fine-tune (base={base_label}, run={run_name})")
        status = 'train_error'
        try:
            proc = subprocess.Popen(
                cmd, cwd=_REPO_ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            for line in proc.stdout:
                print(f"[Training:{sid}] {line.rstrip()}")
            proc.wait()
            if proc.returncode == 0 and os.path.exists(weights_local):
                status = 'trained'
                storage_path = supabase_sync.upload_model_weights(
                    class_name=class_name,
                    weights_path=Path(weights_local),
                    version=version,
                )
                if model_id:
                    supabase_sync.update_model_status(
                        model_id, status='ready',
                        weights_local_path=os.path.relpath(weights_local, _REPO_ROOT),
                        weights_storage_path=(f"model-weights/{storage_path}" if storage_path else None),
                    )
            else:
                if model_id:
                    supabase_sync.update_model_status(model_id, status='failed')
            print(f"[Training:{sid}] finished with status: {status}")
        except Exception as e:
            print(f"[Training:{sid}] error: {e}", file=sys.stderr)
            if model_id:
                supabase_sync.update_model_status(model_id, status='failed')
        finally:
            with _capture_lock:
                session['training_status'] = status
            try:
                session['cap'].release()
            except Exception:
                pass
    threading.Thread(target=_train, daemon=True).start()
```

- [ ] **Step 2: Smoke-check the file still imports**

```bash
python -c "import backend.server"
```

Expected: no traceback.

- [ ] **Step 3: Commit**

```bash
git add backend/server.py
git commit -m "Wrap _trigger_training with user_models insert + weights upload"
```

---

## Task 11: Backfill CLI `scripts/backfill_supabase.py`

One-shot CLI for the existing `anthony_taylor_advil` data and `runs/user_tuned_1/weights/best.pt`.

**Files:**
- Create: `scripts/backfill_supabase.py`
- Create: `scripts/__init__.py` (so `python -m scripts.backfill_supabase` works if needed; one-line empty file)

- [ ] **Step 1: Create `scripts/__init__.py`**

```bash
mkdir -p scripts
```

Create `scripts/__init__.py` with empty content.

- [ ] **Step 2: Create `scripts/backfill_supabase.py`**

```python
"""
scripts/backfill_supabase.py — one-shot backfill of existing local data to Supabase.

Examples:
    # Single class
    python scripts/backfill_supabase.py --class-name anthony_taylor_advil

    # Single class + register an existing weights file as user_models row
    python scripts/backfill_supabase.py \\
        --class-name anthony_taylor_advil \\
        --include-weights runs/user_tuned_1/weights/best.pt \\
        --base-model v10

    # Every directory under user_bottles/ that contains an images/ subfolder
    python scripts/backfill_supabase.py --all
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as a plain script: ensure repo root is on sys.path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

# Load .env if available so SUPABASE_* vars are populated
try:
    from dotenv import load_dotenv
    load_dotenv(_REPO / ".env")
except ImportError:
    pass

from backend import supabase_sync


def _backfill_class(class_name: str, dataset_root: Path) -> int:
    dataset_dir = dataset_root / class_name
    if not (dataset_dir / "images").is_dir():
        print(f"[Backfill] skip {class_name}: no images/ subdir")
        return 0
    uploaded, failed = supabase_sync.upload_dataset_images(class_name, dataset_dir)
    print(f"[Backfill] {class_name}: {uploaded} uploaded, {failed} failed")
    return failed


def _backfill_weights(class_name: str, weights_path: Path, base_model: str) -> int:
    if not weights_path.exists():
        print(f"[Backfill] weights not found: {weights_path}", file=sys.stderr)
        return 1
    version = supabase_sync.compute_next_version(class_name)
    model_id = supabase_sync.insert_model_row(
        class_name=class_name,
        base_model=base_model,
        dataset_path=str(Path("user_bottles") / class_name),
        version=version,
        status="training",
    )
    storage_path = supabase_sync.upload_model_weights(
        class_name=class_name,
        weights_path=weights_path,
        version=version,
    )
    if storage_path and model_id:
        supabase_sync.update_model_status(
            model_id, status="ready",
            weights_local_path=str(weights_path.relative_to(_REPO)) if weights_path.is_absolute() else str(weights_path),
            weights_storage_path=f"model-weights/{storage_path}",
        )
        return 0
    if model_id:
        supabase_sync.update_model_status(model_id, status="failed")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill local user_bottles + weights to Supabase.")
    ap.add_argument("--class-name", help="Class folder under user_bottles/ to upload.")
    ap.add_argument("--all", action="store_true",
                    help="Process every directory under user_bottles/ with an images/ subdir.")
    ap.add_argument("--dataset-root", default="user_bottles",
                    help="Root folder containing class subfolders (default: user_bottles).")
    ap.add_argument("--include-weights", default=None,
                    help="Path to a best.pt to register under user_models for --class-name.")
    ap.add_argument("--base-model", default="v10",
                    help="Label for the base model used to fine-tune (default: v10).")
    args = ap.parse_args()

    if not args.class_name and not args.all:
        ap.error("provide --class-name or --all")

    dataset_root = (_REPO / args.dataset_root) if not Path(args.dataset_root).is_absolute() else Path(args.dataset_root)

    failures = 0
    if args.all:
        for child in sorted(dataset_root.iterdir()):
            if child.is_dir():
                failures += _backfill_class(child.name, dataset_root)
    else:
        failures += _backfill_class(args.class_name, dataset_root)

    if args.include_weights:
        if not args.class_name:
            ap.error("--include-weights requires --class-name")
        weights = Path(args.include_weights)
        if not weights.is_absolute():
            weights = _REPO / weights
        failures += _backfill_weights(args.class_name, weights, args.base_model)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Smoke-check the script's `--help`**

```bash
python scripts/backfill_supabase.py --help
```

Expected: argparse help text appears, no traceback.

- [ ] **Step 4: Commit**

```bash
git add scripts/__init__.py scripts/backfill_supabase.py
git commit -m "Add backfill_supabase script for existing local datasets and weights"
```

---

## Task 12: End-to-end verification (manual)

This task does not modify code. It verifies the full pipeline against the live Supabase project and backfills the existing `anthony_taylor_advil` data.

- [ ] **Step 1: Apply schema migration in Supabase**

Open Supabase Dashboard → SQL Editor → New query. Paste the contents of `supabase/schema.sql` (the entire file is idempotent). Run it.

Verify:
- `bottle_images` now has columns `user_identifier`, `medication_name`, `class_name`, `split` (Dashboard → Table Editor → bottle_images → Columns).
- `user_models` table exists.
- `model-weights` bucket exists (Dashboard → Storage). If not, click "New bucket" → name `model-weights`, public off.

- [ ] **Step 2: Run the full Python test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass (existing tests + new `test_supabase_sync.py` + `test_class_run_dir.py`).

- [ ] **Step 3: Backfill the existing dataset**

```bash
python scripts/backfill_supabase.py \
    --class-name anthony_taylor_advil \
    --include-weights runs/user_tuned_1/weights/best.pt \
    --base-model v10
```

Expected log lines (order may vary):

```
[Sync] client connected to https://...supabase.co
[Sync] uploading 24 images for anthony_taylor_advil…
[Sync] ✓ uploaded 24/24 (0 failed) in N.Ns
[Backfill] anthony_taylor_advil: 24 uploaded, 0 failed
[Sync] inserted user_models row id=… status=training v1
[Sync] uploading weights → model-weights/anthony_taylor_advil/v1/best.pt (XX.X MB)
[Sync] ✓ weights uploaded (XX.X MB)
[Sync] user_models id=… → status=ready
```

- [ ] **Step 4: Verify rows in Supabase Dashboard**

- Table Editor → `bottle_images`: 24 new rows, `class_name=anthony_taylor_advil`, mix of `split=train` and `split=val`.
- Table Editor → `user_models`: 1 new row, `model_name=anthony_taylor_advil`, `status=ready`, `version=1`, `weights_storage_path=model-weights/anthony_taylor_advil/v1/best.pt`.
- Storage → `bottle-images`: folder `anthony_taylor_advil/{train,val}/` with the jpgs.
- Storage → `model-weights`: `anthony_taylor_advil/v1/best.pt` present.

- [ ] **Step 5: Optional — exercise the live capture flow**

(Requires the ESP32 stream and mobile app.) Start the backend (`python backend/server.py`), capture a fresh class through the app, watch the log for `[Sync]` and `[Training]` lines, then re-check the Dashboard for the new rows.

- [ ] **Step 6: No commit needed** (verification-only task)

---

## Self-review notes

- Spec coverage: schema (Task 1), per-class run naming (Tasks 2-3), image upload (Tasks 4-6), model registration + weights upload (Tasks 7-8), capture-pipeline wiring (Task 9), training wrapper (Task 10), backfill (Task 11), verification + backfill execution (Task 12). All spec sections accounted for.
- Type/method consistency: `parse_class_name`, `compute_next_version`, `insert_model_row`, `update_model_status`, `upload_dataset_images`, `upload_model_weights`, `_BUCKET_IMAGES`, `_BUCKET_WEIGHTS`, `class_run_dir`, `--run-name` — all referenced names appear in their defining tasks before being consumed elsewhere. Storage paths consistent (`<class>/<split>/<file>` and `<class>/v<n>/best.pt`) across upload + backfill. Status enum values consistent with schema CHECK constraint (`training`/`ready`/`failed`).
- No placeholders: every code step shows complete code; every command step shows the exact command and expected output.
