# KBeacon → Medication Pipeline Handoff

## Project at a glance

- **Repo:** `C:\Users\Anthony\Documents\CS490\code`
- **Branch:** `mark-app`
- **Plan file:** `docs/superpowers/plans/2026-04-28-kbeacon-medication-pipeline.md` (canonical spec — Tasks 5–8 contain full code blocks ready to execute)
- **Goal:** When the ESP32 detects the KBeacon, the backend resolves the patient (Anthony Taylor), fetches their medications filtered to label codes `{A, B, D, F}`, hands the `label_code → medication_id` map to the YOLO detection subprocess, and on hand-bottle overlap marks the matching medication as `Taken` in `medication_logs`.
- **Architecture:** ESP32 firmware POSTs `/trigger` → backend resolves patient via `PATIENT_CODE` env → fetches meds → spawns `test_with_hand_recognition.py` with `PILLTEK_LABEL_MAP` env var → detection hydrates `CLASS_TO_MED_ID` from `model.names` + the env map → POSTs `/detection-event` with `medication_id` → backend upserts `medication_logs.status='Taken'`.

## Critical decisions to preserve

1. **One ESP32, one patient.** The user explicitly dropped beacon-MAC-based patient routing mid-execution. Patient identity comes from the `PATIENT_CODE` env var via the existing `_resolve_patient_id()` in `backend/server.py`. **Do NOT add `_resolve_patient_by_beacon` or any `patients.beacon_mac` query.** This is also captured in `~/.claude/projects/.../memory/feedback_no_speculative_multi_device.md`.
2. **`beacon_mac` in payload is logged only.** The firmware may include it in the `/trigger` JSON; it's stored in `detection_events.raw_meta` for visibility but does not affect routing.
3. **Allowed labels:** `{A, B, D, F}` — encoded as a CHECK constraint on `medications.label_code` and a `frozenset` in `backend/pipeline_context.py:ALLOWED_LABELS`. If a future trained model adds more letters, both places need updating.
4. **No Claude co-author trailer on commits.** Repo convention.

## Done (Tasks 1, 2, 4)

| # | Subject | Commit | Notes |
|---|---|---|---|
| 1 | Add `medications.label_code` column + index | `6f32cee` | `supabase/schema.sql` migration block dated `2026-04-28`, nullable text, CHECK A/B/D/F. |
| 2 | `backend/pipeline_context.py` (pure module) | `cc97401` | Exposes `ALLOWED_LABELS`, `build_label_med_map`, `model_class_to_med_id`. Also adds empty `backend/__init__.py`. 6 tests in `tests/test_pipeline_context.py`. |
| 3 | (DROPPED) | — | Beacon-MAC patient lookup. Removed from plan; see decision #1 above. Recorded in plan revision commit `56676fe`. |
| 4 | `_fetch_allowed_medications(patient_id)` helper | `a1d79f5` | In `backend/server.py` lines 97–114, plus an import block at lines 68–73 mirroring the `supabase_sync` fallback pattern. Test file `tests/test_server_trigger.py` created with `fake_db` fixture and 2 tests. |

Other relevant commits on this branch:
- `036534e` — merged `beacon_trigger.py` into `backend/server.py` (deleted the duplicate). The current `/trigger` route lives in server.py and is what Task 5 will replace.
- `e8289b0` / `56676fe` — plan revisions.

## Remaining (Tasks 5, 6, 7, 8)

**The plan file has full code blocks for each remaining task.** A subagent or human can implement each one verbatim. The summary below is for orientation only.

### Task 5 — Enrich `/trigger` to pass medication context

**Files to touch:**
- `backend/server.py` — replace the entire `/trigger` route. Add `import json` near the top.
- `tests/test_server_trigger.py` — append `test_trigger_beacon_near_passes_label_map_via_env` and `test_trigger_invalid_event_returns_400`.

**Behavior:** On `beacon_near`, call `_resolve_patient_id()` → `_fetch_allowed_medications(patient_id)` → spawn `test_with_hand_recognition.py` with the spawned-process env containing:
- `PILLTEK_LABEL_MAP` (JSON-serialized `{label_code: medication_id}`)
- `PILLTEK_PATIENT_ID` (only when patient_id is non-None)
- `PILLTEK_BACKEND` (set if not already present, defaults to `http://127.0.0.1:$BACKEND_PORT`)

The CLI flags `--source` and `--weights` should still be passed (they already are after the merge commit). The `beacon_mac` field in the payload should be parsed and logged into `detection_events.raw_meta`, but **not** used for routing.

**Test fixture pattern (already in the file):** the test sets `server._patient_id = 'pid-1'` directly to bypass the cached lookup, then mocks `db.table().select().eq().execute()` to return medication rows.

**Commit message:** `Pass per-patient medication label map to detection via env`

### Task 6 — Hydrate `CLASS_TO_MED_ID` in `test_with_hand_recognition.py`

**Files to touch:**
- `test_with_hand_recognition.py` — replace the empty `CLASS_TO_MED_ID` block at the top of the file (currently lines 26–32 in the working tree) with a `hydrate_class_to_med_id(model_names)` function. Call it in `run_infer` immediately after the YOLO model loads.
- `tests/test_detection_label_map.py` — new file with 3 tests using `importlib.reload` and `patch.dict(os.environ, ...)` to verify the env-var hydration.

**Behavior:**
- Read `PILLTEK_LABEL_MAP` JSON from env at import time (`{'A': '<med-uuid>', ...}`).
- After `model = YOLO(args.weights)`, call `hydrate_class_to_med_id(model.names)`. The function uses the same trailing-letter regex as `pipeline_context.py` (`r'\b([A-Za-z])\s*$'`) to extract the letter from each YOLO class name and looks up the medication UUID. It populates the module-global `CLASS_TO_MED_ID` as a side effect.
- In `post_detection_event`, also include `patient_id` in the payload when `_PATIENT_ID = os.environ.get('PILLTEK_PATIENT_ID')` is set.

**Commit message:** `Hydrate CLASS_TO_MED_ID from PILLTEK_LABEL_MAP env var`

### Task 7 — `/pipeline-debug` endpoint

**Files to touch:**
- `backend/server.py` — add a new POST route just before `/health` that runs `_resolve_patient_id()` + `_fetch_allowed_medications()` and returns the resolved context as JSON. **Does not spawn any subprocess.** Useful for verifying Supabase data is shaped correctly without needing the ESP32 powered on.
- `tests/test_server_trigger.py` — append `test_pipeline_debug_returns_resolved_context`.

**Commit message:** `Add /pipeline-debug endpoint for hardware-free trigger verification`

### Task 8 — Full test suite + smoke verification

- `pytest tests/ -v` — should all pass. Only the previously-skipped test in `test_capture_headless.py` remains skipped.
- Live smoke test of `/pipeline-debug` against real Supabase (curl recipe is in the plan).
- Optional: live `/trigger` smoke against a powered ESP32 with the beacon nearby (logs in the plan).

## Subagent execution recipe (for resuming)

If continuing in a new session with subagent-driven development:

1. Read this handoff and the plan file.
2. Recreate the todo list — Tasks 5, 6, 7, 8 only (1, 2, 4 are done; 3 is deleted).
3. Per task, dispatch:
   - One general-purpose implementer (sonnet for Task 5/6, haiku for Task 7/8) with the full task text from the plan and a "Context" section reminding it of the constraints in **Critical decisions to preserve** above.
   - One general-purpose spec reviewer (haiku) — verify against plan text.
   - One `superpowers:code-reviewer` (haiku) — code quality. Block on Critical or Important issues; minor suggestions are advisory.

The plan file already contains exact code blocks, exact test names, and exact commit messages. Implementers should not improvise.

## Files in flight at handoff time

Working tree is **clean for everything related to this plan**. Modified-but-unrelated files from prior work in this branch are listed in `git status` but were not touched by Tasks 1–4. They include `app/login.tsx`, `package.json`, `working_server/.../cam_and_server_code.ino.ino`, etc. — leave them alone.

## Risks / open items

- **Trained model class-name convention.** The hydration logic assumes class names look like `Bottle A`, `Bottle B`, `Bottle D`, `Bottle F` (trailing letter). The current model in `runs/detect/runs/train_v10/weights/best.pt` should be sanity-checked: `python -c "from ultralytics import YOLO; print(YOLO('runs/detect/runs/train_v10/weights/best.pt').names)"`. If the names don't end in single letters from the allowed set, `hydrate_class_to_med_id` will return `{}` and detection runs untagged.
- **`label_code` UI is not in scope.** A caregiver currently has no app screen to set a medication's `label_code`. For now it must be set via SQL update in Supabase Studio. A future task can add this to the medication-entry screen.
- **Caregiver notification.** The user's original request mentioned reflecting "Taken" status to the caretaker. Today the caregiver sees updates by re-querying `medication_logs`. A real-time push (Supabase realtime channel subscription on the caregiver's app session) is a separate task, not in this plan.
- **`_PATIENT_CODE` must be set before the server starts.** If unset, `_resolve_patient_id()` returns `None`, `_fetch_allowed_medications` returns `{}`, and detection runs but never marks anything taken. `/pipeline-debug` is the fastest way to confirm the env var is correctly wired.
