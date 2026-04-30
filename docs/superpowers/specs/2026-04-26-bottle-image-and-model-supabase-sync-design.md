# Bottle Image + Model Supabase Sync — Design

**Date:** 2026-04-26
**Status:** Approved by user

## Problem

The bottle capture pipeline saves images locally to `user_bottles/<class_name>/` and triggers fine-tuning on those images, but nothing reaches Supabase:

- Captured images are not inserted into the `bottle_images` table or uploaded to the `bottle-images` storage bucket from the Python pipeline. (The frontend has an `uploadBottleImage()` helper, but it is never invoked by the capture flow.)
- Trained model weights are never persisted off-disk.
- There is no record of which user/medication a trained model corresponds to.
- Run folders are named `runs/user_tuned_<N>/` — non-descriptive and prone to confusion across users/medications.

The current `anthony_taylor_advil` dataset (24 images) and its trained `runs/user_tuned_1/weights/best.pt` are not in Supabase.

## Goal

After capture completes for any `<user>_<medication>` class:

1. All captured images are uploaded to Supabase Storage and indexed in `bottle_images`.
2. Fine-tuning runs automatically (already does — wiring stays).
3. The resulting `best.pt` is uploaded to Supabase Storage and a `user_models` row is recorded.
4. Run folders are named after the class so they don't collide across users/medications.
5. Existing local-only data (`anthony_taylor_advil` + `user_tuned_1`) can be backfilled with a one-shot script.

## Non-goals

- Reworking the mobile app's `uploadBottleImage()` path. That flow stays as-is.
- RLS / multi-tenant security. Schema follows the existing prototype convention (RLS disabled).
- Real-time progress reporting of uploads to the mobile app. Logs only, for now.

## Architecture

```
backend/
  server.py              (existing — wire upload + post-train hook)
  supabase_sync.py       (NEW — all Supabase I/O, fault-tolerant)
capture_bottles.py       (existing — change run-dir naming only)
scripts/
  backfill_supabase.py   (NEW — one-shot: anthony_taylor_advil + user_tuned_1)
supabase/
  schema.sql             (extend bottle_images; add user_models; add bucket)
```

`supabase_sync.py` is the **only** Python module that talks to Supabase from the capture/training pipeline. `server.py` and the backfill script call into it. This keeps the network/IO concerns out of the capture and training code paths.

## Schema changes

Append to `supabase/schema.sql` (idempotent — safe to re-run):

```sql
-- ── Extend bottle_images (backward-compatible) ───────────────────────────────
ALTER TABLE bottle_images ALTER COLUMN medication_id DROP NOT NULL;
ALTER TABLE bottle_images ADD COLUMN IF NOT EXISTS user_identifier text;
ALTER TABLE bottle_images ADD COLUMN IF NOT EXISTS medication_name text;
ALTER TABLE bottle_images ADD COLUMN IF NOT EXISTS class_name      text;
ALTER TABLE bottle_images ADD COLUMN IF NOT EXISTS split           text
  CHECK (split IN ('train','val'));

-- ── user_models ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_models (
  id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_identifier      text NOT NULL,
  medication_name      text NOT NULL,
  model_name           text NOT NULL,    -- = class_name (e.g. anthony_taylor_advil)
  base_model           text,             -- e.g. 'v10' or full path string
  dataset_path         text,             -- local dataset directory
  weights_local_path   text,             -- runs/<class_name>[_vN]/weights/best.pt
  weights_storage_path text,             -- model-weights/<class_name>/v<N>/best.pt
  version              integer DEFAULT 1,
  status               text DEFAULT 'ready'
                            CHECK (status IN ('training','ready','failed')),
  created_at           timestamptz DEFAULT now()
);
ALTER TABLE user_models DISABLE ROW LEVEL SECURITY;

-- ── model-weights storage bucket ─────────────────────────────────────────────
-- May fail if run as a non-service-role user. Fallback: create manually in
-- Dashboard → Storage → New bucket. Name: model-weights, Public: false.
INSERT INTO storage.buckets (id, name, public)
  VALUES ('model-weights', 'model-weights', false)
  ON CONFLICT DO NOTHING;
```

Existing `bottle-images` bucket is unchanged.

## Run folder naming

`capture_bottles.py` currently calls `next_run_dir(runs_dir, prefix="user_tuned_")`, producing `runs/user_tuned_1`, `runs/user_tuned_2`, ... Replace with a class-name-aware helper:

```python
def class_run_dir(runs_dir: Path, class_name: str) -> Path:
    base = runs_dir / class_name
    if not base.exists():
        return base
    i = 2
    while (runs_dir / f"{class_name}_v{i}").exists():
        i += 1
    return runs_dir / f"{class_name}_v{i}"
```

- First run for a class → `runs/anthony_taylor_advil/`
- Retraining the same class → `runs/anthony_taylor_advil_v2/`, `_v3/`, ...
- Old `runs/user_tuned_*/` folders are left alone.

The `next_run_dir` helper is kept for backward compatibility but no longer called from the main path.

## Capture → upload flow

The capture endpoint `/capture-frame` (in `backend/server.py`) currently flips `session['status'] = 'done'` and calls `_trigger_training(session)` once `captures_done >= total`. Insert the upload between those steps:

```
[capture_frame: last frame saved]
    → session.status = 'done'
    → background thread A: supabase_sync.upload_dataset_images(class_name, dataset_dir)
    → background thread B: _trigger_training(session)   (existing)
```

A and B run concurrently. Training does not wait for upload — they touch different resources. The HTTP response returns immediately; both threads are daemonized.

`upload_dataset_images(class_name, dataset_dir)`:

1. Walks `<dataset_dir>/images/{train,val}/*.jpg`.
2. For each file:
   - Upload bytes to `bottle-images/<class_name>/<split>/<filename>` (upsert=true).
   - Insert `bottle_images` row: `{class_name, user_identifier, medication_name, split, filename, storage_path}`.
3. Per-file try/except — one failure doesn't abort the batch. Failed paths are retried once after a 2-second pause.
4. Logs `[Sync] uploading N images for <class_name>…` then `[Sync] ✓ uploaded X/N (Y failed)`.

Parsing `class_name` → `(user_identifier, medication_name)`: split on the **last** underscore. Examples:
- `anthony_taylor_advil` → user=`anthony_taylor`, med=`advil`
- `bob_the_builder_tylenol_500mg` → user=`bob_the_builder`, med=`tylenol_500mg`

This is a documented assumption. Class names that don't fit (e.g. single-token names) fall back to `user_identifier=class_name`, `medication_name=''`, and a warning is logged.

## Train → save → upload flow

Wrap, don't replace, the existing subprocess training. `_trigger_training()` becomes:

```python
def _trigger_training(session):
    def _run():
        class_name = session['class_name']
        # 1. Determine version from existing user_models rows (max(version)+1, default 1)
        # 2. Insert user_models row with status='training', weights paths null
        # 3. subprocess.run([...capture_bottles.py --class-name <c> --train-only])
        # 4. On exit code 0:
        #      - Locate best.pt at runs/<class_name>[_vN]/weights/best.pt
        #      - upload to model-weights/<class_name>/v<version>/best.pt
        #      - Update user_models row: status='ready', weights_local_path, weights_storage_path
        #    On exit code != 0:
        #      - Update user_models row: status='failed'
        # 5. session['training_status'] = 'trained' | 'train_error'
    threading.Thread(target=_run, daemon=True).start()
```

The subprocess is preserved — it's the proven path and isolates the heavy ultralytics import from the Flask process. Only the post-process Supabase work is new.

The base model used for the fine-tune is recorded as `base_model='v10'` (the basename of the directory `runs/detect/runs/train_v10/`), since that's what `YOLO_WEIGHTS` resolves to by default. If `YOLO_WEIGHTS` env var is overridden, store the full path string verbatim.

## Backfill script (`scripts/backfill_supabase.py`)

One-shot CLI for existing local data:

```
python scripts/backfill_supabase.py --class-name anthony_taylor_advil
python scripts/backfill_supabase.py --all
python scripts/backfill_supabase.py --class-name anthony_taylor_advil \
    --include-weights runs/user_tuned_1/weights/best.pt --base-model v10
```

- Calls the same `supabase_sync.upload_dataset_images()` and a new `supabase_sync.upload_model_weights()`.
- `--all` iterates every directory under `user_bottles/` that contains an `images/` subfolder.
- `--include-weights <path>` registers an existing weights file under `user_models` (model_name = class_name, version = next available, status='ready').

## supabase_sync.py public API

```python
get_client() -> Client | None              # cached, env-driven (SUPABASE_URL/SUPABASE_SERVICE_KEY)
upload_dataset_images(class_name: str, dataset_dir: Path) -> tuple[int, int]
    # returns (uploaded, failed)
upload_model_weights(
    class_name: str,
    weights_path: Path,
    base_model: str,
    dataset_path: Path,
    version: int | None = None,            # auto-computed if None
) -> str | None                            # returns storage path, or None on failure
update_model_status(model_id: str, status: str, **fields) -> None
insert_model_row(class_name, base_model, dataset_path, version, status='training') -> str
    # returns the new row's id
```

Each function is fault-tolerant: it logs and returns a sentinel rather than raising, except where the caller specifically wants to detect failure (e.g. backfill exit code).

## Logging convention

All output prefixed `[Sync]` for Supabase I/O, `[Train]` for training. Examples:

```
[Sync] uploading 24 images for anthony_taylor_advil…
[Sync] ✓ uploaded 24/24 (0 failed) in 3.1s
[Sync] inserted user_models row id=… status=training
[Train:abc12345] starting fine-tune (base=v10, output=runs/anthony_taylor_advil/)
[Train:abc12345] finished with exit code 0
[Sync] uploading weights → model-weights/anthony_taylor_advil/v1/best.pt
[Sync] ✓ weights uploaded (14.7 MB)
[Sync] user_models id=… → status=ready
```

Failures still log a single line and never crash the calling thread.

## Assumptions

1. **Class-name parsing.** `<user>_<medication>` split on the **last** underscore. Documented in `supabase_sync.py` and the backfill script.
2. **Bucket creation via SQL.** May fail under the dashboard's anon role; fallback is manual creation in the Storage UI. The `schema.sql` header notes both paths.
3. **`medication_id` stays optional.** The mobile app's `uploadBottleImage` flow continues to set it; the backend pipeline leaves it null. Both shapes coexist in the table.
4. **Idempotency.** Storage uploads use `upsert=true`. DB inserts are not deduped — a re-run of the backfill creates duplicate `bottle_images` rows. Acceptable for the prototype; a `UNIQUE (class_name, filename, split)` constraint can be added later if needed.
5. **Weights size.** `best.pt` is ~15-20 MB, well under Supabase storage limits. Uploaded as `application/octet-stream`.
6. **Concurrency.** Image upload and training run as separate background threads after capture completes. They touch different resources (DB+storage vs. local disk + GPU/CPU), so no coordination is required.

## End-to-end flow (after this change)

1. Mobile app calls `POST /start-capture` with `class_name=anthony_taylor_advil`.
2. Backend opens ESP32 stream; mobile app calls `POST /capture-frame` 24 times.
3. On the 24th frame:
   - `session.status = 'done'`.
   - Thread A: `upload_dataset_images()` syncs all 24 jpgs to `bottle-images/anthony_taylor_advil/{train,val}/...`, inserts 24 `bottle_images` rows.
   - Thread B: `_trigger_training()` inserts a `user_models` row (`status='training'`), spawns `capture_bottles.py --train-only`, which fine-tunes from `train_v10/best.pt` and writes to `runs/anthony_taylor_advil/weights/best.pt`. On success, the wrapper uploads weights to `model-weights/anthony_taylor_advil/v1/best.pt` and updates the row to `status='ready'`.
4. Mobile app polls `GET /capture-status/<sid>` and sees `training_status='trained'` once the model row is `ready`.
5. The backfill script handles the existing `anthony_taylor_advil` images + `user_tuned_1` weights as a one-time catch-up.
