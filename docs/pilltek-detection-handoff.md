# PillTek Detection Pipeline — Handoff (2026-04-30)

This handoff captures the state of the **ESP32 → backend → detection → mobile**
pipeline after a long debugging session. Read this top-to-bottom before touching
anything; many of the comments in the code reference subtleties that took hours
to discover.

If you only have 30 seconds: open issues are listed in the **"Open issues"**
section. Everything above it is context.

---

## What PillTek is

A medication-adherence prototype for a single patient (Anthony Taylor) that:

1. A **KBeacon BLE tag** is attached to the patient (or their pill organizer).
2. An **ESP32-CAM** scans for the beacon. When RSSI ≥ -70 dBm, the camera
   becomes "near" → POSTs `beacon_near` to the Flask backend.
3. The **backend** (`backend/server.py`) spawns a Python detection subprocess
   (`test_with_hand_recognition.py`). YOLO finds bottles, MediaPipe finds the
   hand, and when a hand overlaps a bottle for ~half of the last 16 frames at
   ≥80 % YOLO confidence, the script POSTs `hand_bottle_overlap` back to the
   backend.
4. The backend writes `medication_logs.status = 'Taken'` and mirrors that
   onto `medications.status` so the mobile/web UI cards reflect today's
   outcome. A daily reset job flips `medications.status` back to `'Pending'`
   at local midnight.
5. The **Expo/React Native app** (`app/`) reads from Supabase and shows the
   patient/caregiver views.

There is exactly one ESP32 and one patient (no multi-device routing). The
patient is identified by the env var `PATIENT_CODE` on the backend; the
detection script receives the resolved UUID via `PILLTEK_PATIENT_ID`.

---

## Repo layout — only the parts that matter

```
code/
├── backend/
│   ├── server.py             ← Flask backend, all HTTP endpoints
│   ├── pipeline_context.py   ← Pure helpers (label↔med-id, class↔med-id)
│   ├── supabase_sync.py      ← Dataset image upload + model row helpers
│   └── requirements.txt
├── working_server/
│   └── cam_and_server_code.ino/
│       ├── cam_and_server_code.ino.ino  ← MAIN ESP32 SKETCH (BLE + camera)
│       ├── wifi_provisioning.ino        ← NVS + portal Wi-Fi setup
│       └── board_config.h               ← Camera pin defs
├── test_with_hand_recognition.py ← Live YOLO + MediaPipe subprocess
├── capture_bottles.py            ← Headless dataset capture + fine-tune
├── capture_user_bottles.py       ← Older interactive capture (still used)
├── supabase/schema.sql           ← Single source of truth for DB schema
├── runs/                         ← YOLO weights live under runs/<run_name>/weights/best.pt
│   └── detect/runs/train_v10/weights/best.pt   ← default weights
├── user_bottles/                 ← Per-patient datasets for fine-tunes
├── app/                          ← Expo Router screens (React Native)
│   ├── patient-home.js, patient-detail.js, patient-medications.js
│   ├── add-medication.js     ← writes medications row + label dropdown
│   ├── capture-bottles.js    ← drives /start-capture session
│   └── ...
├── lib/
│   ├── api.js                ← Supabase CRUD helpers used by app
│   ├── medication-store.js, patient-store.js, ...
│   └── supabase.js           ← Client init
├── tests/                    ← pytest, all green at last commit
│   ├── test_pipeline_context.py
│   ├── test_server_trigger.py        ← incl. /pipeline-debug
│   ├── test_detection_label_map.py
│   └── test_capture_headless.py
├── docs/
│   ├── pilltek-detection-handoff.md  ← this file
│   ├── kbeacon-pipeline-handoff.md
│   ├── capture-bottles-fps-handoff.md
│   └── superpowers/plans/             ← plan docs from past tasks
└── .env                       ← (gitignored) backend + app secrets
```

`runs/`, `user_bottles/`, `__pycache__/`, `node_modules/` are gitignored.

---

## How the pieces talk to each other

```
┌──────────────┐         BLE scan
│ KBeacon tag  │ ─────────────────────► ESP32-CAM
└──────────────┘                        │
                                        │ POST /trigger {event:"beacon_near"|"beacon_far"}
                                        ▼
                                ┌─────────────────────┐
                                │  backend/server.py  │
                                │  Flask, port 5000   │
                                └─────────────────────┘
                                        │
       ┌────────────────────────────────┤
       │ subprocess.Popen               │ Supabase (REST)
       │ env: PILLTEK_LABEL_MAP,        ▼
       │      PILLTEK_PATIENT_ID  ┌─────────────────────┐
       ▼                          │  patients,           │
┌──────────────────────────┐      │  medications,        │
│ test_with_hand_recognition.py │  │  medication_logs,    │
│  YOLO + MediaPipe        │      │  detection_events,   │
│  http://192.168.0.31:81  │ ◄──── stream MJPEG          │  bottle_images,      │
│         /stream          │      │  user_models         │
└──────────────────────────┘      └─────────────────────┘
       │ POST /detection-event           ▲
       │ {event_type, bottle_class,      │ Supabase (REST,
       │  bottle_label, conf,            │ EXPO_PUBLIC_*)
       │  medication_id, patient_id}     │
       └────────────────────────────────►┘
                                        │
                                ┌─────────────────────┐
                                │   Expo / React Native│
                                │   app/*              │
                                └─────────────────────┘
```

### Backend endpoints (`backend/server.py`)

| Endpoint | Purpose |
|---|---|
| `POST /trigger` | ESP32 webhook: `beacon_near` spawns detection subprocess; `beacon_far` terminates it. Both paths log to `detection_events`. |
| `POST /detection-event` | Detection script reports `hand_bottle_overlap`. Upserts `medication_logs` and mirrors `'Taken'` onto `medications.status`. |
| `POST /start-capture` | Mobile app starts a headless dataset-capture session. Kills detection first to free the stream. |
| `GET /capture-preview/<session_id>` | Streams the current ESP32 frame as JPEG. |
| `POST /capture-frame` | Saves a single annotated capture to disk + dataset. |
| `GET /capture-status/<session_id>` | Polls capture progress; returns `training_status` when fine-tune kicks in. |
| `POST /pipeline-debug` | Diagnostic: runs the patient + label-map lookup that `/trigger` does, without spawning detection. |
| `POST /reset-daily-status` | Manually triggers the medication-status reset (cron-safe). |
| `GET /health` | `{supabase, patient_code, patient_id, detection_running}` |
| `GET /stream-status/<sid>`, `/debug-frame/<sid>` | One-off diagnostics |

### IPC: backend → detection script

The backend passes context via env vars when spawning the subprocess:

```
PILLTEK_LABEL_MAP   '{"A":"<med-uuid>","D":"<med-uuid>","F":"<med-uuid>"}'
PILLTEK_PATIENT_ID  '7b9b66ce-2877-4947-9f56-443daca61551'
PILLTEK_BACKEND     'http://127.0.0.1:5000'
```

The detection script reads these once at startup, hydrates `CLASS_TO_MED_ID`
from YOLO `model.names` (extracts the trailing letter of each class name —
"Bottle A" → "A"), and maps it to the medication UUID. **Allowed labels are
`{A, B, D, F}`** — anything else is silently skipped.

---

## Database schema

`supabase/schema.sql` is the canonical schema. **Run any new migrations there,
then in the Supabase SQL editor.** Production database currently has:

- `profiles, patients, medications, medication_logs, bottle_images, detection_events, user_models`
- `medications.label` (A/B/D/F) — added by frontend dropdown PR; the backend
  reads this column to build the YOLO-class → medication map
- `medications.status` — mirrors today's `medication_logs` outcome
- Storage buckets: `bottle-images`, `model-weights`

If you add a column, **also run** `NOTIFY pgrst, 'reload schema'` in the
Supabase SQL editor — PostgREST caches the schema and the app will throw
`Could not find the X column ... in the schema cache` until you do.

---

## How to run everything (from a fresh clone)

```bash
# 1. Install deps
pip install -r backend/requirements.txt
npm install

# 2. Fill in .env (gitignored)
SUPABASE_URL=...
SUPABASE_SERVICE_KEY=...        # service role (server-side ONLY)
EXPO_PUBLIC_SUPABASE_URL=...
EXPO_PUBLIC_SUPABASE_ANON_KEY=...
PATIENT_CODE=PTK-XXXX           # must match patients.patient_code in DB
ESP32_STREAM_URL=http://192.168.0.31:81/stream
YOLO_WEIGHTS=runs/detect/runs/train_v10/weights/best.pt
EXPO_PUBLIC_BACKEND_URL=http://192.168.0.X:5000
EXPO_PUBLIC_FIREBASE_*=...

# 3. Run the backend
python backend/server.py

# 4. Run the mobile app
npx expo start
# (use --offline if api.expo.dev DNS is flaky)

# 5. Flash the ESP32
# Open working_server/cam_and_server_code.ino in Arduino IDE,
# Tools → Board → ESP32S3 Dev Module, Tools → PSRAM → OPI PSRAM
# Upload. The first boot serves a Wi-Fi provisioning portal at
# its AP SSID, set credentials there. Subsequent boots load NVS.
```

---

## Open issues (read these before making changes)

### 1. Stream FPS degrades over time
**Severity:** medium — affects detection accuracy when stream drops below ~5 fps.

The ESP32-CAM serves MJPEG over HTTP at ~30 fps when polled by a *plain*
browser. When `test_with_hand_recognition.py` consumes it via
`cv2.VideoCapture(url, CAP_FFMPEG)` (the `ThreadedVideoCapture` class), the
stream starts at ~25-40 fps and degrades to ~0 fps over a couple of minutes.

**Diagnosis already done in the FPS counter:** `[Loop]` shows the main loop
holds at 60+ fps the whole time, but `stream` fps decays. So the loop is fast
— it's just looking at the same cached frame repeatedly because the
background reader is stalling.

**Why the obvious fix didn't work:** Backend uses `MjpegCapture` (a
requests-based reader) for the same stream and works fine for `/capture-preview`.
We tried porting it into the detection script as `MjpegCapture` (still in the
file, currently unused). With the detection script as the only client it
*flatlined at 1 fps* and forced reconnects every 10 s. We have no clean
explanation for why the same class behaves differently in the same network
conditions when it's the detection script's reader vs. the backend's. We
reverted to `ThreadedVideoCapture`. Both classes are in the file at lines
~243 and ~175 respectively.

**Hypotheses (untested):**
- ESP32-CAM only allows one concurrent stream client. If a stale
  detection process or an open browser tab holds the connection, the new
  client gets throttled. Always check `Get-Process python` for stragglers
  before testing.
- ESP32 BLE/Wi-Fi coexistence — the chip handles BLE scan callbacks on the
  same core that runs the camera HTTP server. Heavy BLE traffic can starve
  the stream. The Arduino sketch calls `pBLEScan->stop()` for 1 s after a
  near event, which helps, but doesn't eliminate the problem.
- ESP32 heap fragmentation. The sketch prints `Free heap: N` every 5 s in
  `loop()` — watch this during a long run. If it trends downward, the
  stream side has a leak somewhere in `app_httpd.cpp` or
  `cam_and_server_code.ino`.

**Suggested next step:** add a main-loop watchdog that abandons the
`ThreadedVideoCapture` and rebuilds it when `cap.frame_id` hasn't advanced
in ~3 s. The `frame_id` field already exists on both capture classes for
exactly this purpose.

### 2. YOLO model needs more training (user-handled)
The current weights at `runs/detect/runs/train_v10/weights/best.pt` produce
false-positive bottle detections at high confidence. The user is retraining
with more data; you don't need to do anything here. Once new weights land
they'll go into the same path or be set via the `YOLO_WEIGHTS` env var.

### 3. Hand-detection threshold (just changed)
`test_with_hand_recognition.py:494` — `min_hand_detection_confidence` and
`min_tracking_confidence` are now both **0.5** (previously 0.4). 0.5 is
strict enough that MediaPipe shouldn't produce phantom hands on bottles or
arms, while still catching a real hand reliably.

### 4. ESP32 BLE state desync (recovery in place)
There is a long history of bugs with the ESP32 BLE scanner getting wedged
when the backend was unreachable. The current sketch handles this with two
heartbeats (in `working_server/.../cam_and_server_code.ino.ino:330+`):

- **Far heartbeat**: while `cameraActive == true` and the beacon has been
  silent for `MISSING_GRACE_MS = 60 s`, re-POSTs `beacon_far` every
  `FAR_HEARTBEAT_MS = 60 s`.
- **Near heartbeat**: while `beaconCurrentlyNear == true`, re-POSTs
  `beacon_near` every `NEAR_HEARTBEAT_MS = 60 s`. Recovers state when the
  backend is restarted with the beacon already near.
- BLE scan resume after the 1 s near-pause is **unconditional** (used to
  be gated on `cameraActive`, which left the scanner stuck off when
  near-POSTs failed).

Don't reintroduce a `cameraActive` gate on the BLE-resume — that's exactly
what caused the wedge.

---

## Recent fixes from this session (so you don't re-debug)

Pinned in commit history:

- **`cb55eec`** — Tied event POSTs to the same overlap decision as the
  visualization. Before, `hand_overlaps_bottle()` was called twice per
  frame (once in `draw_hands`, once in event firing), each call
  double-appending to `_grab_history`, and the event firing reported the
  *highest-confidence* bottle in frame instead of the bottle the hand was
  actually touching. Result: events for Bottle F when the hand was on
  Bottle B. Now: smoothed decision is computed once per hand, returns
  `(overlapping, cls, conf)`, drawing and POST share it.
- **`eecb320`** — Renamed `medications.label_code` → `medications.label`
  to match the frontend column added in the teammate's add-medication PR.
- **`4946504`** — Added `/pipeline-debug` for hardware-free trigger
  verification.

Uncommitted at the time of writing:

- `EVENT_BOTTLE_CONF_THRESHOLD = 0.80` gate (suppresses POST when YOLO
  conf < 80%, even if the hand-overlap smoother says yes)
- `medications.status` mirroring + `_daily_reset_loop` (resets to
  `'Pending'` at local midnight via daemon thread, plus
  `POST /reset-daily-status` for manual cron)
- ESP32 sketch heartbeats + BLE-resume fix described above
- `_resolve_patient_id` uses `.limit(1)` instead of `.maybe_single()`
  (newer supabase-py versions return `None` from `maybe_single()` when no
  rows match, breaking `res.data`)

Run `git diff` to see them before you touch anything.

---

## Gotchas / non-obvious things

1. **`PATIENT_CODE` must match a row** in the `patients` table. If it
   doesn't, `_resolve_patient_id` logs `[Supabase] No patient row found
   for PATIENT_CODE=...` and the detection script gets `PILLTEK_LABEL_MAP=
   '{}'`, so events fire but never mark anything taken. Verify with
   `curl -X POST http://127.0.0.1:5000/pipeline-debug -d '{}'`.
2. **The ESP32 stream supports exactly one client.** If `/capture-preview`
   sessions weren't released, or a browser tab is open, the detection
   subprocess gets a degraded connection. The backend's `/start-capture`
   handler kills detection AND releases existing capture sessions before
   opening a new one — don't bypass that.
3. **The 16-frame `_grab_history` smoother is per-`hand_idx`.** When
   MediaPipe loses and reacquires a hand, it can re-use index 0, picking
   up stale history. With `num_hands=1` (current) this is mostly fine.
4. **`detect_for_video` requires monotonically increasing timestamps.**
   The script computes `int(time.time() * 1000) - start_ms`. Don't switch
   to `time.monotonic()` alone — MediaPipe wants epoch-ms-ish numbers.
5. **The backend's `_db.table(...).update(...)` requires a filter.** When
   resetting all medications to `'Pending'`, we use `.in_('status',
   ['Taken','Missed'])` instead of an unfiltered update.
6. **CHECK constraint with NULL is satisfied** — `status IN (...)` returns
   NULL when status is NULL, which Postgres treats as not-violated. So a
   row with `status=NULL` passes the check; don't rely on the constraint
   to enforce non-null.
7. **No Claude co-author trailer on commits in this repo.** Per the user.
8. **Don't add multi-device infrastructure.** The system is single-ESP32,
   single-patient by design — no beacon-MAC routing, no per-device session
   IDs.
9. **YOLO class indexing is fragile.** Class 0 is `'-'` (a dummy/empty
   class). Real bottles start at index 1. A new training run with a
   different class list would silently break the label-letter extraction.
   Check `model.names` after retraining.

---

## How to verify the system end-to-end

```bash
# 1. Backend up?
curl http://127.0.0.1:5000/health
# {"status":"ok","supabase":true,"patient_code":"PTK-XXXX","patient_id":"...","detection_running":false}

# 2. Patient + label map resolved?
curl -X POST http://127.0.0.1:5000/pipeline-debug -d '{}' -H 'Content-Type: application/json'
# {"patient_id":"...","label_map":{"A":"...","D":"...","F":"..."}}

# 3. Trigger a fake near (no ESP32 needed)
curl -X POST http://127.0.0.1:5000/trigger \
     -d '{"event":"beacon_near"}' -H 'Content-Type: application/json'
# Backend logs: [Pipeline] starting detection: patient=... labels=[...]

# 4. Stop it
curl -X POST http://127.0.0.1:5000/trigger \
     -d '{"event":"beacon_far"}' -H 'Content-Type: application/json'

# 5. Run the test suite
pytest tests/ -v
# 41 passed, 1 skipped (the desktop-only capture test)
```

---

## Where to start if you're new

1. Read `working_server/cam_and_server_code.ino/cam_and_server_code.ino.ino`
   end-to-end. The comments are dense for a reason — every block is
   responding to a specific failure mode that has been seen.
2. Read `backend/server.py` from `_resolve_patient_id` down through
   `/trigger`, `/detection-event`, and `_daily_reset_loop`. That's the
   whole control plane.
3. Read `test_with_hand_recognition.py` from `hand_overlaps_bottle`
   through the main loop in `run_infer`. The smoother + event-firing logic
   has been rewritten three times in this session; the comments explain
   why.
4. If you're touching the database, edit `supabase/schema.sql` AND run
   the change in the Supabase SQL editor + `NOTIFY pgrst, 'reload schema'`.
5. If you're touching the ESP32, do not introduce blocking work in the
   BLE callback — set flags, handle in `loop()`. The current sketch is
   the result of multiple rounds of removing blocking code from the
   callback.

Good luck.
