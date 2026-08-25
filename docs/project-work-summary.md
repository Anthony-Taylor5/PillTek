# PillTek: Description of Work Completed

PillTek is a vision-based medication adherence system built as a CSUN capstone
project. A BLE beacon wakes an ESP32-S3 camera when the patient approaches the
medication area; the camera streams video to a Python backend that runs a custom
YOLO detector for pill bottles and MediaPipe for hands; when a hand overlaps a
bottle the event is written to Supabase and surfaces in an Expo/React Native app
as a medication marked `Taken`.

The work below is grouped by area. Repository: 60 commits across three
contributors (Anthony Taylor, Mark Youssef, Mahsa).

---

## 1. Dataset Construction and Hand Annotation (Roboflow)

All training data was captured, labeled, and versioned by hand. Nothing was
scraped or reused from a public dataset.

**Capture.** `take_pics.py` was written to drive dataset collection directly off
the ESP32 MJPEG stream: a live window with single-keypress saving into the
correct class folder (`a` -> Bottle A, `b` -> Bottle B, `c` -> Bottle C,
`d` -> combined scene, `e` -> hazard, `f` -> hand, `x` -> same-bottle scene).
Capturing through the real camera rather than a phone meant the training
distribution matched deployment: same lens, same quality-12 JPEG compression,
same lighting and mounting angle. `grab_frames.py` was a smaller companion used
to pull five frames and sanity-check image quality before a capture session.

**Volume.** 824 source images were collected across seven folders:

| Folder | Images |
|---|---|
| A images | 103 |
| B images | 103 |
| C images | 100 |
| combined images | 214 |
| hazard images | 103 |
| hand images | 100 |
| same bottle images | 101 |

**Annotation.** Every image was hand-labeled in Roboflow (workspace
`seniordesign-yht0c`, project `seniordesign-hf6ds`). Bounding boxes were drawn
manually for each bottle instance, hazard, and combined scene. This was the
single most labor-intensive part of the project: five separate annotation
passes, since each new dataset version required labeling the newly captured
images and re-checking the older ones as the class list expanded.

**Versioning.** The dataset was grown in five deliberate stages, each exported
from Roboflow as its own YOLO26-format release. The class list expanded as
failure modes were found in the field:

| Local dir | Roboflow ver | Classes (nc) | Train / Valid / Test | What it added |
|---|---|---|---|---|
| `v2_with_background` | 7 | 3 | 375 / 122 / 58 | Bottles A, B, C only |
| `v4_with_combined` | 9 | 4 | 630 / 142 / 68 | Explicit background/negative class `-`, multi-bottle scenes |
| `v6_with_hazards` | 10 | 5 | 1281 / 133 / 67 | `hazard` class for dangerous bottle combinations |
| `v8_with_hands` | 11 | 5 | 1491 / 153 / 77 | Hand-in-frame images so bottles stay detected under occlusion |
| `v10_with_same_bottles` | 12 | 7 | 1698 / 172 / 87 | Bottle D and Bottle F for visually identical bottle disambiguation |

The final release is 1,957 images. Preprocessing was auto-orientation with EXIF
stripping. Roboflow augmentation produced three versions of each source image:
random crop 0-25 percent, rotation -15 to +15 degrees, brightness -25 to +25
percent, exposure -15 to +15 percent, Gaussian blur 0-2.5 px, and salt-and-pepper
noise on 0.73 percent of pixels. The augmentation profile was chosen to mirror
real ESP32 artifacts: the sensor is noisy and the mount shifts slightly, so
noise and small rotations were weighted more heavily than color jitter.

`create_augments.py` was also written as an offline OpenCV/PIL augmentation path
(rotation, flips, crop, color jitter, blur, hue/saturation shift, class parsed
from filename) for tripling the dataset locally without a Roboflow round trip.

The `-` background class in v4 was added specifically to fix false positives:
the v2 model was firing on cabinet edges and other cylindrical objects. Adding
explicitly labeled negative scenes suppressed that. Similarly, v10's Bottle D
and F classes exist because two real prescriptions ship in the identical amber
bottle and the v8 model could not tell them apart, so extra images were captured
and labeled with the distinguishing cap and label detail.

---

## 2. YOLO Model Development

**Training pipeline.** `train_two_datasets_yolo26n.py` implements staged
transfer learning across all five dataset versions. Training starts from the
`yolo26n.pt` nano base weights and fine-tunes forward: v2 -> v4 -> v6 -> v8 ->
v10, each stage initializing from the previous stage's `best.pt` rather than
from scratch. `--dataset both` runs the whole chain sequentially, which is how
the production weights at `runs/detect/runs/train_v10/weights/best.pt` were
produced. The script validates the dataset layout up front (`data.yaml` plus
`train/`, `valid/`, `test/`) and supports `--epochs` and `--device` so the same
code runs on a CPU laptop or a GPU box.

Nano was picked deliberately over larger YOLO variants: the pipeline runs on a
consumer laptop alongside MediaPipe on a second thread, and the accuracy gain
from a bigger backbone did not justify the frame-rate loss on low-resolution
ESP32 frames.

**Evaluation tooling.** `infer_dataset_images_to_csv.py` runs the trained model
over an image folder and writes a per-image CSV: `image_name`,
`bottles_detected`, `hand_present`, `interaction`, `hazard_flag`, full
`raw_detections` JSON, and `timestamp`. This produced `inference_log.csv` and
made it possible to diff model versions quantitatively instead of eyeballing the
live feed. `test_inference.py` is a stripped YOLO-only live runner used to
smoke-test new weights without paying the MediaPipe cost.

**Per-user fine-tuning.** Beyond the shared model, `capture_bottles.py` and
`capture_user_bottles.py` implement an in-product fine-tune loop: the patient
photographs their own bottles through the app, images land in `user_bottles/`,
and a fine-tune run is kicked off automatically against those images. Run
folders are named per class via a `class_run_dir` helper with a `--run-name`
override, validated to be a single path component so a crafted class name cannot
escape the runs directory.

---

## 3. MediaPipe Hand Detection and Interaction Logic

MediaPipe Hand Landmarker (`hand_landmarker.task`, downloaded automatically on
first run) runs entirely independently of YOLO on its own thread. Detecting
hands with MediaPipe rather than adding a "hand" class to YOLO was a deliberate
split: MediaPipe gives 21 landmarks per hand, so interaction can be decided on
fingertip positions instead of a coarse bounding box that overlaps a bottle
whenever an arm merely passes in front of it.

**Interaction rule.** An interaction is flagged when any hand landmark falls
inside a bottle bounding box (with 10 px padding) for a majority of the last 16
frames. The temporal majority vote was necessary because single-frame overlap
fired constantly on reach-past motions. A separate confidence gate,
`EVENT_BOTTLE_CONF_THRESHOLD = 0.80`, suppresses events when YOLO is not
confident about which bottle it is looking at, which prevents logging the wrong
medication.

**Smoothing and display.** Bottle boxes are averaged over the last five frames
to kill jitter from the noisy sensor. Color coding communicates state at a
glance: gray background class, green Bottle A, blue Bottle B, yellow Bottle C,
purple hazard, red hand with no overlap, orange hand actively interacting.

---

## 4. ESP32 Firmware

The firmware in `working_server/cam_and_server_code.ino/` is roughly 2,900 lines
across the main sketch, the HTTP stream server, Wi-Fi provisioning, and pin
configuration. It does three things concurrently on a Seeed XIAO ESP32-S3 Sense:
BLE scanning, MJPEG camera streaming, and HTTP event notification.

**BLE proximity.** The sketch scans continuously for a KBeacon at MAC
`dd:34:02:0a:2d:f1`. RSSI at or above -70 dBm means near, below means far.
Critically, the BLE callback only sets volatile flags; the actual work happens
in the main loop. Doing HTTP POSTs inside the BLE callback blocked the BLE stack
and degraded scan timing badly, which was one of the harder bugs to isolate.

**State handling.** Near/far state is driven purely off current RSSI rather than
off the `cameraActive` flag, because `cameraActive` can desync from the backend
after a failed POST or an ESP32 reset. Both a near heartbeat and a far heartbeat
re-POST periodically so the backend can recover from a missed edge. BLE scan
resume after the one-second near-pause was made unconditional after a bug where
the scan silently never restarted.

**Camera.** `CAMERA_GRAB_LATEST` is required; `CAMERA_GRAB_WHEN_EMPTY` caused
multi-second stalls because the driver would serve stale queued frames. JPEG
quality is pinned at 12 to keep frames small enough for the Wi-Fi link. PSRAM
must be enabled (OPI PSRAM) or the camera will not initialize at all. NimBLE
v1.4.2 is the pinned version; v2.3.6 broke compatibility.

**Wi-Fi provisioning.** `wifi_provisioning.ino` adds NVS-backed credential
storage with an AP-mode captive portal fallback, so the board no longer needs
credentials hardcoded and reflashed for every network change. This mattered
because the setup moved between a home router and a phone hotspot.

**Events.** On near: activate camera, `POST /trigger {"event":"beacon_near"}`.
On far: stop camera, `POST /trigger {"event":"beacon_far"}`. The stream is
served at `http://<ESP32_IP>:81/stream`.

---

## 5. Backend

`backend/server.py` is the Flask backend and the hub of the system. It began as
a standalone `beacon_trigger.py` and was merged into a single server (`036534e`)
so that beacon triggering, detection events, capture sessions, and Supabase
logging all live in one process.

**Endpoints:**

| Endpoint | Purpose |
|---|---|
| `POST /trigger` | ESP32 webhook. `beacon_near` spawns the detection subprocess, `beacon_far` terminates it. Both log to `detection_events`. |
| `POST /detection-event` | Detection script reports `hand_bottle_overlap`. Upserts `medication_logs` and mirrors `Taken` onto `medications.status`. |
| `POST /start-capture` | App starts a headless dataset-capture session. Kills detection first to free the single-client ESP32 stream. |
| `GET /capture-preview/<sid>` | Proxies the current ESP32 frame as a JPEG for the app's live preview. |
| `POST /capture-frame` | Saves one annotated capture to disk and to the dataset. |
| `GET /capture-status/<sid>` | Polls capture progress, returns `training_status` once fine-tuning starts. |
| `POST /pipeline-debug` | Runs the same patient and label-map resolution `/trigger` does, without spawning anything. |
| `POST /reset-daily-status` | Manual trigger for the daily medication-status reset. |
| `GET /health` | Reports Supabase connectivity, patient code, resolved patient id, detection-running state. |
| `GET /stream-status/<sid>`, `GET /debug-frame/<sid>` | One-off stream diagnostics. |

**Patient and medication resolution.** `_resolve_patient_id()` maps the
`PATIENT_CODE` env var to a patient UUID (cached after first lookup, using
`.limit(1)` rather than `.maybe_single()`, which threw on zero rows).
`_fetch_allowed_medications()` pulls that patient's medications filtered to the
allowed slot labels `{A, B, D, F}`.

**Process context passing.** `backend/pipeline_context.py` is a pure,
dependency-free module holding `ALLOWED_LABELS`, `build_label_med_map`, and
`model_class_to_med_id`. The backend builds a `label -> medication_id` map and
hands it to the spawned detection subprocess through environment variables
(`PILLTEK_LABEL_MAP` as JSON, `PILLTEK_PATIENT_ID`, `PILLTEK_BACKEND`). The
detection script hydrates `CLASS_TO_MED_ID` by extracting the trailing letter
from each YOLO class name in `model.names` ("Bottle A" -> "A") and matching it
against the map. This keeps the vision code free of any database dependency: it
does not know what Supabase is, it just posts events carrying a medication id it
was handed.

An earlier design routed patients by beacon MAC address. That was deliberately
removed (`56676fe`) once scope settled on one ESP32 and one patient, rather than
carrying speculative multi-device infrastructure.

**Supabase sync.** `backend/supabase_sync.py` handles dataset and model
artifacts: `upload_dataset_images` with per-file retry, `compute_next_version`
for model versioning, `insert_model_row` / `update_model_status`, and
`upload_model_weights` into the `model-weights` bucket. A `local_only` model
status was added for the real case where training succeeds but the storage
upload fails, so the weights stay usable rather than the run being marked
failed. `scripts/backfill_supabase.py` retroactively pushes existing local
datasets and weights into Supabase.

**Daily reset.** `_daily_reset_loop()` flips `medications.status` back to
`Pending` at local midnight so the app's cards represent today, not the last
time a bottle was touched.

---

## 6. Database

`supabase/schema.sql` is the single source of truth, 149 lines covering seven
tables and two storage buckets:

- `profiles` mirrors Firebase Auth users with a role of caregiver, patient, or self.
- `patients` holds caregiver linkage, a shareable `patient_code` (e.g. `PTK-A1B2`), and DOB/phone.
- `medications` holds name, dosage, frequency, a `times` JSONB array, refill info, `status`, and the `label` slot letter constrained to A/B/D/F with an index on `(patient_id, label)`.
- `medication_logs` is one row per dose occurrence per day, with a `UNIQUE (medication_id, log_date, scheduled_time)` constraint so the detection pipeline and manual confirmation cannot double-log. `source` distinguishes `manual`, `esp32`, and `detection`.
- `bottle_images` tracks captured photos and their storage paths, extended to carry `user_identifier`, `medication_name`, `class_name`, and a train/val `split` for the fine-tune pipeline while staying backward compatible with the app's original rows.
- `detection_events` is the raw event log: event type, YOLO class id, label, confidence, patient, and a `raw_meta` JSONB blob.
- `user_models` is the per-user trained model registry with base model, dataset path, local and storage weight paths, version, and a status of training/ready/local_only/failed.

Buckets: `bottle-images` and `model-weights`, both private. RLS is disabled
throughout, documented in the schema as a prototype decision with a note to add
firebase_uid-matched policies before production.

---

## 7. Mobile Application

An Expo Router / React Native app (Expo 54, RN 0.81.5, React 19, new
architecture enabled) with 28 screens under `app/`, supporting two distinct
flows.

**Caregiver flow:** add and manage patients, per-patient medication lists,
medication detail views, a caregiver calendar, patient daily logs, and log entry
detail.

**Patient / self flow:** patient home dashboard, own medications, schedule,
profile, daily logs, and log views.

**Auth:** Firebase Auth for login, account creation, password reset, and email
verification, with the profile row mirrored into Supabase. A patient links to a
caregiver by entering the caregiver-generated patient code.

**Medication entry:** add-medication writes the medications row, including a
time picker for multiple daily doses and a slot-label dropdown (A/B/D/F) that
ties the medication to a YOLO class. That dropdown replaced an earlier flow that
sent users straight into bottle capture (`2162ab4`).

**Bottle capture screen:** `app/capture-bottles.js` drives a `/start-capture`
session and shows a live preview proxied from the ESP32 through the backend.
This screen took substantial debugging work:

- Fabric clips all children of a view that combines `borderRadius` with `overflow: hidden`, which made the entire camera feed invisible on Android despite correct layout bounds. Fixed by removing `overflow: hidden` and moving the radius to the inner image (`9828db3`).
- React Native's built-in `Image` resets to a Fresco `EmptyDrawable` when `source.uri` changes, producing a black flash between every frame. Fixed by switching to `expo-image` with a stable `recyclingKey` (`90b3767`).
- When the ESP32 returned a byte-identical JPEG, React's setState bailout suppressed the render, `onLoad` never fired, and the polling loop stalled. Fixed with a last-frame comparison that re-polls immediately instead of waiting.
- Concurrent fetches caused Fresco to cancel decodes mid-flight. Fixed by pacing the loop on `onLoad` with a one-second safety-timer fallback.
- Preview work was then moved off the JS thread and server overhead trimmed (`da5a656`).

**State and data layer:** `lib/api.js` wraps all Supabase CRUD;
`medication-store.js`, `patient-store.js`, `med-detail-store.js`, and
`role-store.js` hold client state; `schedule-utils.js` handles dose scheduling
math. `lib/supabase.js` guards against an unconfigured environment by returning
`null` instead of throwing, so the app degrades to in-memory stores rather than
crashing when `.env` is missing.

---

## 8. Integration

Making the five subsystems act as one system was its own body of work.

**The full path:** KBeacon RSSI crosses threshold -> ESP32 POSTs `beacon_near`
-> backend resolves the patient from `PATIENT_CODE`, fetches medications
filtered to allowed labels, spawns the detection subprocess with the label map
in its environment -> detection pulls the MJPEG stream, runs YOLO and MediaPipe
on separate threads, detects a sustained hand-bottle overlap above 80 percent
confidence -> POSTs `hand_bottle_overlap` with a `medication_id` -> backend
upserts `medication_logs.status = 'Taken'` and mirrors it onto
`medications.status` -> the app reads it from Supabase and updates the card.

**Threading architecture in the detection script.** Three components run
independently so no stage blocks another: `ThreadedVideoCapture` reads frames
into a size-1 queue, always keeping the newest and dropping stale ones, with
auto-reconnect when the stream drops; `PillWorker` runs YOLO on a persistent
background thread; `HandWorker` runs MediaPipe on another. The main loop draws
whatever results already exist and never waits. This is why the display stays
smooth even when inference is slower than the incoming stream.

**Latency work.** `latency_diagnostic.py` instruments every stage separately
(capture, queue wait, YOLO, MediaPipe) and writes per-frame CSV plus an
aggregated summary. Queue depth was the key metric: a growing queue means
inference is falling behind the stream. This tooling identified Wi-Fi
instability as the dominant source of latency spikes, not model inference, and
confirmed that `CAMERA_GRAB_LATEST` on the firmware side was worth several
seconds.

**Resource contention.** The ESP32 camera server accepts exactly one stream
client. Detection and capture both want it, so `/start-capture` explicitly kills
the running detection subprocess before opening its own connection. Stale
connections that outlive their client are a known cause of FPS decay over time.

**Testability without hardware.** `/pipeline-debug` and the ability to POST a
synthetic `beacon_near` mean the entire backend-to-database path can be verified
with the ESP32 unplugged, which made iteration far faster than power-cycling the
board for each test.

**Tests.** A pytest suite under `tests/` covers the pipeline context helpers,
the trigger route and its label-map passing, the detection script's env-var
hydration, the headless capture flow, run-directory naming, and Supabase sync,
at 41 passing and 1 skipped (a desktop-only capture test).

---

## 9. Deployment and Operations

**Backend.** Runs as `python backend/server.py`, Flask threaded on
`0.0.0.0:5000`, configured entirely through environment variables loaded from a
gitignored `.env`: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` (service role, server
side only), `PATIENT_CODE`, `BACKEND_HOST`, `BACKEND_PORT`, `ESP32_STREAM_URL`,
and `YOLO_WEIGHTS`. Dependencies are pinned in `backend/requirements.txt`.

**Firmware.** Flashed from Arduino IDE with the board set to XIAO_ESP32S3 and
PSRAM set to OPI PSRAM. First boot serves the provisioning portal on its own AP;
Wi-Fi credentials are then persisted in NVS, so moving to a new network needs no
reflash.

**Mobile.** `npx expo start` for development, `expo run:android` for a native
build. `usesCleartextTraffic` is enabled on Android because the backend and
ESP32 are plain HTTP on the LAN. Public config reaches the app through
`EXPO_PUBLIC_*` variables (Supabase URL and anon key, backend URL, Firebase
config).

**Database migrations.** Applied by running the relevant block from
`supabase/schema.sql` in the Supabase SQL editor. A hard-won operational note is
documented: after adding a column you must run `NOTIFY pgrst, 'reload schema'`,
or PostgREST serves a cached schema and the app fails with "Could not find the X
column in the schema cache."

**Scheduled work.** The daily status reset runs in-process on a background loop
and is also exposed at `POST /reset-daily-status` so it can be driven by an
external cron instead.

**Documentation.** Three handoff documents in `docs/` capture the system state
in enough detail for a cold start: `pilltek-detection-handoff.md` (full
architecture, endpoint table, gotchas, end-to-end verification procedure),
`kbeacon-pipeline-handoff.md` (the beacon-to-medication task breakdown and the
decisions to preserve), and `capture-bottles-fps-handoff.md` (the preview
rendering investigation with root causes for each symptom). Design specs and
implementation plans live under `docs/superpowers/`.

---

## 10. Known Limitations

- Stream FPS degrades over a long session, attributed to a combination of stale stream clients, ESP32 BLE/Wi-Fi radio coexistence, and heap fragmentation. Free heap is printed every five seconds for monitoring.
- The YOLO model still benefits from more training data, particularly for visually similar bottles.
- RLS is disabled on all Supabase tables. This is a prototype decision that must be reversed before any real deployment.
- Wi-Fi stability, especially on a phone hotspot, is the dominant remaining source of latency spikes.
- The capture preview runs below the desired frame rate on the Android emulator with software GPU; the identified next step is pointing `expo-image` at the backend URL directly to eliminate the JS-side base64 round trip.
