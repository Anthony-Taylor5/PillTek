"""
backend/server.py — PillTek backend server.

Extends beacon_trigger.py with:
  • Supabase event logging for every beacon + detection event
  • POST /detection-event  — receives structured events from test_with_hand_recognition.py
  • POST /trigger          — unchanged ESP32 beacon webhook (backward compatible)
  • POST /start-capture    — launches capture_user_bottles.py for a medication
  • GET  /capture-status/<session_id> — polls capture progress

Run:
  cd /path/to/PillTek
  python backend/server.py

Environment variables (copy .env.example → .env and fill in):
  SUPABASE_URL          — your Supabase project URL
  SUPABASE_SERVICE_KEY  — service role key (bypasses RLS; keep server-side only)
  PATIENT_CODE          — patient_code of the patient whose beacon this server monitors
  BACKEND_HOST          — bind address (default 0.0.0.0)
  BACKEND_PORT          — port (default 5000)
  ESP32_STREAM_URL      — default ESP32 camera stream URL for capture sessions
  YOLO_WEIGHTS          — path to YOLO weights used for capture fine-tuning
"""

import os
import subprocess
import sys
import time
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import requests as _requests
from flask import Flask, request, jsonify, Response

# ── Load .env if present (python-dotenv is optional) ─────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    pass  # python-dotenv not installed — rely on shell environment

# ── Supabase client ───────────────────────────────────────────────────────────
try:
    from supabase import create_client, Client as SupabaseClient
    _supabase_url = os.environ.get('SUPABASE_URL', '')
    _supabase_key = os.environ.get('SUPABASE_SERVICE_KEY', '')
    if _supabase_url and _supabase_key:
        _db: SupabaseClient = create_client(_supabase_url, _supabase_key)
        print(f'[Supabase] Connected to {_supabase_url}')
    else:
        _db = None
        print('[Supabase] SUPABASE_URL / SUPABASE_SERVICE_KEY not set — DB logging disabled.')
except Exception as _e:
    _db = None
    print(f'[Supabase] Client init failed: {_e}')

# ── Supabase sync (dataset images + model weights) ───────────────────────────
try:
    from backend import supabase_sync
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    import supabase_sync  # type: ignore[no-redef]

# ── Patient lookup ────────────────────────────────────────────────────────────
_PATIENT_CODE = os.environ.get('PATIENT_CODE', '').strip().upper()
_patient_id: str | None = None  # UUID resolved on first use


def _resolve_patient_id() -> str | None:
    """Look up the patient UUID by PATIENT_CODE once, then cache it."""
    global _patient_id
    if _patient_id:
        return _patient_id
    if not _db or not _PATIENT_CODE:
        return None
    try:
        res = _db.table('patients').select('id').eq('patient_code', _PATIENT_CODE).maybe_single().execute()
        if res.data:
            _patient_id = res.data['id']
            print(f'[Supabase] Resolved patient_id={_patient_id} for code={_PATIENT_CODE}')
    except Exception as e:
        print(f'[Supabase] Patient lookup failed: {e}')
    return _patient_id


def _log_event(event_type: str, **kwargs) -> None:
    """Insert a detection_events row.  Never raises — logs failures to stderr."""
    if not _db:
        return
    try:
        row = {
            'event_type':   event_type,
            'patient_id':   _resolve_patient_id(),
            'triggered_at': datetime.now(timezone.utc).isoformat(),
            **{k: v for k, v in kwargs.items() if v is not None},
        }
        _db.table('detection_events').insert(row).execute()
    except Exception as e:
        print(f'[Supabase] _log_event failed ({event_type}): {e}', file=sys.stderr)


def _upsert_med_log(medication_id: str, scheduled_time: str | None, status: str) -> None:
    """Insert / update a medication_logs row from a detection event."""
    if not _db or not _resolve_patient_id():
        return
    try:
        today = datetime.now(timezone.utc).date().isoformat()
        row = {
            'medication_id':  medication_id,
            'patient_id':     _resolve_patient_id(),
            'log_date':       today,
            'scheduled_time': scheduled_time,
            'taken_at':       datetime.now(timezone.utc).isoformat() if status == 'Taken' else None,
            'status':         status,
            'source':         'detection',
        }
        _db.table('medication_logs').upsert(
            row,
            on_conflict='medication_id,log_date,scheduled_time'
        ).execute()
    except Exception as e:
        print(f'[Supabase] _upsert_med_log failed: {e}', file=sys.stderr)


# ── MjpegCapture — requests-based MJPEG reader ───────────────────────────────
# Replaces cv2.VideoCapture(url, CAP_FFMPEG) for ESP32 MJPEG HTTP streams.
# CAP_FFMPEG stalls on fresh connections (2-10 s delay or never resolves).
# requests handles plain HTTP far more reliably: connects in <1 s and
# reconnects automatically on dropped streams.

class MjpegCapture:
    def __init__(self, url: str):
        self._url     = url
        self._frame   = None          # latest decoded BGR numpy array
        self._lock    = threading.Lock()
        self._stopped = False
        self._thread  = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self):
        # ESP32-CAM keeps the TCP socket alive between JPEGs (multipart boundaries
        # dribble in), so requests' per-read timeout alone does not catch a stall.
        # We track the wall-clock time of the last decoded frame and force a
        # reconnect if no new frame has arrived in STALL_SECONDS.
        STALL_SECONDS = 3.0
        while not self._stopped:
            try:
                # (connect=5, read=3): a silent socket trips read-timeout too.
                with _requests.get(self._url, stream=True, timeout=(5, 3)) as r:
                    buf     = b''
                    last_ok = time.monotonic()
                    for chunk in r.iter_content(chunk_size=4096):
                        if self._stopped:
                            return
                        buf += chunk
                        if time.monotonic() - last_ok > STALL_SECONDS:
                            raise TimeoutError(
                                f'no new JPEG in {STALL_SECONDS}s — forcing reconnect'
                            )
                        # Extract every complete JPEG frame from the buffer
                        while True:
                            soi = buf.find(b'\xff\xd8')   # JPEG start-of-image
                            eoi = buf.find(b'\xff\xd9')   # JPEG end-of-image
                            if soi == -1 or eoi == -1 or eoi <= soi:
                                break
                            jpg   = buf[soi:eoi + 2]
                            buf   = buf[eoi + 2:]
                            import cv2 as _cv2, numpy as _np
                            frame = _cv2.imdecode(
                                _np.frombuffer(jpg, dtype=_np.uint8),
                                _cv2.IMREAD_COLOR,
                            )
                            if frame is not None:
                                with self._lock:
                                    self._frame = frame.copy()
                                last_ok = time.monotonic()
            except Exception as exc:
                if not self._stopped:
                    print(f'[MjpegCapture] stream error, reconnecting in 1 s: {exc}')
                    time.sleep(1)

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self):
        self._stopped = True


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

_running_process: subprocess.Popen | None = None

# ── Capture sessions ──────────────────────────────────────────────────────────
_capture_sessions: dict[str, dict] = {}
_capture_lock = Lock()

_ESP32_STREAM_URL = os.environ.get('ESP32_STREAM_URL', 'http://192.168.0.211:81/stream')
_YOLO_WEIGHTS     = os.environ.get('YOLO_WEIGHTS', 'runs/detect/runs/train_v10/weights/best.pt')
_REPO_ROOT        = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))

# ── Capture utilities (cv2 + capture_bottles) ─────────────────────────────────
sys.path.insert(0, _REPO_ROOT)
try:
    import cv2
    import numpy as np
    from capture_bottles import (
        ThreadedCapture, prepare_dataset_dirs, write_data_yaml,
        save_capture, next_run_dir, BOX_PRESETS, TOTAL_CAPTURES,
    )
    _capture_available = True
    print('[Server] Capture ready (cv2 + capture_bottles loaded)')
except Exception as _cap_err:
    cv2 = None
    np  = None
    _capture_available = False
    print(f'[Server] Capture unavailable (cv2/capture_bottles missing): {_cap_err}')


@app.route('/trigger', methods=['POST'])
def trigger():
    """
    ESP32 beacon webhook — unchanged from beacon_trigger.py.
    Additionally logs beacon events to Supabase.
    """
    global _running_process

    data  = request.get_json(silent=True) or {}
    event = data.get('event', '')
    print(f'[ESP32] event: {event}')

    if event == 'beacon_near':
        _log_event('beacon_near', raw_meta={'source': 'esp32'})

        if _running_process is None or _running_process.poll() is not None:
            print('[Server] Starting detection subprocess...')
            _running_process = subprocess.Popen(
                [sys.executable, 'test_with_hand_recognition.py'],
                cwd=os.path.join(os.path.dirname(__file__), '..')
            )
        else:
            print('[Server] Detection subprocess already running.')
        return 'ok'

    elif event == 'beacon_far':
        _log_event('beacon_far', raw_meta={'source': 'esp32'})

        if _running_process is not None and _running_process.poll() is None:
            print('[Server] Stopping detection subprocess...')
            _running_process.terminate()
            try:
                _running_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                print('[Server] Force-killing subprocess...')
                _running_process.kill()
        _running_process = None
        return 'ok'

    else:
        return jsonify({'error': 'invalid event'}), 400


@app.route('/detection-event', methods=['POST'])
def detection_event():
    """
    Receives a structured detection event from test_with_hand_recognition.py.

    Expected JSON body:
    {
      "event_type":   "hand_bottle_overlap" | "bottle_detected",
      "bottle_class": 1,           // YOLO class id
      "bottle_label": "Bottle A",
      "confidence":   0.87,
      "medication_id": "<uuid>",   // optional — if known from the class mapping
      "raw_meta":     {}           // optional extra data
    }
    """
    data = request.get_json(silent=True) or {}

    event_type   = data.get('event_type',   'bottle_detected')
    bottle_class = data.get('bottle_class')
    bottle_label = data.get('bottle_label')
    confidence   = data.get('confidence')
    medication_id = data.get('medication_id')
    raw_meta     = data.get('raw_meta',     {})

    print(f'[Detection] {event_type} | class={bottle_class} ({bottle_label}) conf={confidence:.2f}')

    _log_event(
        event_type,
        bottle_class=bottle_class,
        bottle_label=bottle_label,
        confidence=confidence,
        raw_meta=raw_meta,
    )

    # If a medication_id is provided (mapped from bottle class → medication UUID),
    # mark the medication as Taken for today.
    if medication_id and event_type == 'hand_bottle_overlap':
        _upsert_med_log(
            medication_id=medication_id,
            scheduled_time=None,   # detection doesn't know which time slot
            status='Taken',
        )

    return jsonify({'status': 'logged'}), 200


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


@app.route('/start-capture', methods=['POST'])
def start_capture():
    """
    Initialise a headless capture session driven by the mobile app.

    Expected JSON body:
    {
      "class_name": "john_tylenol_500mg",
      "source":     "http://...:81/stream"   // optional
    }

    Returns: { "session_id": "<uuid>" }
    """
    if not _capture_available:
        return jsonify({'error': 'Capture unavailable — cv2/capture_bottles missing on server'}), 503

    # Stop detection subprocess in background so the request returns immediately
    global _running_process
    if _running_process is not None and _running_process.poll() is None:
        proc_to_stop = _running_process
        _running_process = None
        def _stop_detection():
            print('[Capture] Stopping detection subprocess to free ESP32 stream…')
            proc_to_stop.terminate()
            try:
                proc_to_stop.wait(timeout=4)
            except subprocess.TimeoutExpired:
                proc_to_stop.kill()
        threading.Thread(target=_stop_detection, daemon=True).start()

    # Release every existing session's stream connection before opening a new one.
    # The ESP32 MJPEG server only supports ONE concurrent client. Without this,
    # each app reload leaves the previous MjpegCapture holding the socket, so
    # every subsequent /start-capture gets connection-refused and returns 503 forever.
    with _capture_lock:
        for old_sid, old_session in list(_capture_sessions.items()):
            old_cap = old_session.get('cap')
            if old_cap is not None:
                try:
                    old_cap.release()
                    print(f'[Capture] Released old session {old_sid[:8]}')
                except Exception:
                    pass
        _capture_sessions.clear()

    data       = request.get_json(silent=True) or {}
    class_name = data.get('class_name', 'user_bottle')
    source     = data.get('source', _ESP32_STREAM_URL)

    dataset_dir = Path(_REPO_ROOT) / 'user_bottles' / class_name
    dirs        = prepare_dataset_dirs(dataset_dir)
    data_yaml   = write_data_yaml(dataset_dir, class_name=class_name)

    session_id = str(uuid.uuid4())
    session = {
        'session_id':      session_id,
        'class_name':      class_name,
        'status':          'running',
        'captures_done':   0,
        'total':           TOTAL_CAPTURES,
        'training_status': None,
        'cap':             None,   # set by background thread once stream connects
        'dirs':            dirs,
        'data_yaml':       data_yaml,
    }
    with _capture_lock:
        _capture_sessions[session_id] = session

    # Connect to stream in background — /capture-preview returns 503 until ready
    def _init_cap():
        sid = session_id[:8]
        time.sleep(1.2)  # let detection process fully release the stream
        print(f'[Capture:{sid}] Connecting to ESP32 stream: {source}')
        try:
            cap = MjpegCapture(source)

            # Poll for the first valid frame for up to 8 seconds.
            # A fresh HTTP MJPEG connection from the ESP32 often takes 2-4 s to
            # deliver the first frame — calling cap.read() exactly once after the
            # 1.5 s warmup sleep is too early and returns (False, None), which
            # previously caused the session to be marked error immediately.
            deadline = time.time() + 19.0
            ok, test = False, None
            while time.time() < deadline:
                ok, test = cap.read()
                if ok and test is not None:
                    break
                print(f'[Capture:{sid}] waiting for first frame…')
                time.sleep(0.3)

            if ok and test is not None:
                h, w = test.shape[:2]
                with _capture_lock:
                    session['cap'] = cap
                    session['_preview_waiting_logged'] = False
                print(f'[Capture:{sid}] ✓ Stream ready — first frame {w}×{h}px received')
            else:
                cap.release()
                with _capture_lock:
                    session['status'] = 'error'
                print(
                    f'[Capture:{sid}] ✗ Stream opened but no frames received after warmup.\n'
                    f'              → Check ESP32 is powered on and streaming at {source}\n'
                    f'              → On Windows host: curl {source} to verify reachability'
                )
        except Exception as e:
            with _capture_lock:
                session['status'] = 'error'
            print(
                f'[Capture:{sid}] ✗ Stream connection failed: {e}\n'
                f'              → source={source}\n'
                f'              → Set ESP32_STREAM_URL env var to the correct stream URL'
            )
    threading.Thread(target=_init_cap, daemon=True).start()

    return jsonify({'session_id': session_id}), 200


_CORS_HEADERS = {
    'Access-Control-Allow-Origin':  '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Cache-Control': 'no-cache, no-store, must-revalidate',
    'Pragma': 'no-cache',
}


@app.route('/capture-preview/<session_id>', methods=['GET', 'OPTIONS'])
def capture_preview(session_id):
    """Return the latest frame from the ESP32 stream as a JPEG image."""
    if request.method == 'OPTIONS':
        return Response('', 204, headers=_CORS_HEADERS)
    if not _capture_available:
        print(f'[Preview:{session_id[:8]}] 503 — cv2/capture_bottles unavailable')
        return Response('', 503, headers=_CORS_HEADERS)
    with _capture_lock:
        session = _capture_sessions.get(session_id)
    if not session:
        print(f'[Preview:{session_id[:8]}] 404 — session not found')
        return Response('', 404, headers=_CORS_HEADERS)
    # If _init_cap gave up, tell the frontend immediately (404 stops the retry loop).
    if session.get('status') == 'error':
        return Response('', 503, headers=_CORS_HEADERS)

    cap = session.get('cap')
    if cap is None:
        # still connecting — log once per session to avoid spam
        if not session.get('_preview_waiting_logged'):
            print(f'[Preview:{session_id[:8]}] 503 — stream still connecting to ESP32')
            session['_preview_waiting_logged'] = True
        return Response('', 503, headers=_CORS_HEADERS)
    ok, frame = cap.read()
    if not ok or frame is None:
        return Response('', 503, headers=_CORS_HEADERS)
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])

    # Rate-limited throughput log — one line per second instead of one per
    # frame. Stdout prints + numpy reductions on every request were adding
    # non-trivial per-request latency (esp. on Windows console).
    now = time.monotonic()
    session['_pv_frames'] = session.get('_pv_frames', 0) + 1
    session['_pv_bytes']  = session.get('_pv_bytes', 0) + len(buf)
    last_log = session.get('_pv_last_log', 0.0)
    if now - last_log >= 1.0:
        n   = session['_pv_frames']
        kb  = session['_pv_bytes'] / 1024.0
        dt  = now - last_log if last_log else 1.0
        print(f'[Preview:{session_id[:8]}] {n/dt:.1f} fps, {kb/dt:.0f} KB/s')
        session['_pv_frames'] = 0
        session['_pv_bytes']  = 0
        session['_pv_last_log'] = now
    return Response(buf.tobytes(), mimetype='image/jpeg', headers=_CORS_HEADERS)


@app.route('/stream-status/<session_id>', methods=['GET'])
def stream_status(session_id):
    """Diagnostic: returns whether the ESP32 stream is connected for a session."""
    with _capture_lock:
        session = _capture_sessions.get(session_id)
    if not session:
        return jsonify({'error': 'session not found'}), 404
    cap = session.get('cap')
    frame_ok = False
    if cap is not None:
        ret, frame = cap.read()
        frame_ok = ret and frame is not None
    return jsonify({
        'session_id':    session_id,
        'status':        session['status'],
        'cap_connected': cap is not None,
        'frame_ok':      frame_ok,
        'captures_done': session['captures_done'],
        'esp32_url':     _ESP32_STREAM_URL,
    }), 200


@app.route('/debug-frame/<session_id>', methods=['GET'])
def debug_frame(session_id):
    """Save the current frame to debug_frame.jpg so you can open it on Windows."""
    with _capture_lock:
        session = _capture_sessions.get(session_id)
    if not session:
        return jsonify({'error': 'session not found'}), 404
    cap = session.get('cap')
    if cap is None:
        return jsonify({'error': 'stream not ready'}), 503
    ok, frame = cap.read()
    if not ok or frame is None:
        return jsonify({'error': 'no frame available'}), 503
    path = os.path.join(_REPO_ROOT, 'debug_frame.jpg')
    cv2.imwrite(path, frame)
    stats = {
        'saved_to':  path,
        'shape':     list(frame.shape),
        'mean':      round(float(frame.mean()), 1),
        'min':       int(frame.min()),
        'max':       int(frame.max()),
    }
    print(f'[DebugFrame] {stats}')
    return jsonify(stats), 200


@app.route('/capture-frame', methods=['POST'])
def capture_frame():
    """
    Save the current frame from the stream with a YOLO bounding box annotation.

    Expected JSON body:
    {
      "session_id": "<uuid>",
      "box_type":   "short" | "tall" | "wide"
    }

    Returns: { "captures_done": 5, "total": 24, "status": "running" | "done" }
    """
    if not _capture_available:
        return jsonify({'error': 'Capture unavailable'}), 503

    data       = request.get_json(silent=True) or {}
    session_id = data.get('session_id')
    box_type   = data.get('box_type', 'tall')

    with _capture_lock:
        session = _capture_sessions.get(session_id)
    if not session:
        return jsonify({'error': 'session not found'}), 404
    if session['status'] != 'running':
        return jsonify({'error': f"session is {session['status']}"}), 409

    cap = session.get('cap')
    if cap is None:
        return jsonify({'error': 'stream still connecting, try again in a moment'}), 503
    preset = BOX_PRESETS.get(box_type, BOX_PRESETS['tall'])
    ok, frame = cap.read()
    if not ok or frame is None:
        return jsonify({'error': 'no frame available from stream'}), 503

    fh, fw = frame.shape[:2]
    box_w  = int(fw * preset['w_frac'])
    box_h  = int(fh * preset['h_frac'])
    idx    = session['captures_done']

    save_capture(
        frame=frame,
        box_cx=fw // 2, box_cy=fh // 2,
        box_w=box_w, box_h=box_h,
        dirs=session['dirs'],
        index=idx,
    )

    with _capture_lock:
        session['captures_done'] += 1
        done = session['captures_done']
        if done >= session['total']:
            session['status'] = 'done'

    print(f"[Capture:{session_id[:8]}] {done}/{session['total']} saved (box={box_type})")

    if done >= session['total']:
        _trigger_image_upload(session)
        _trigger_training(session)

    return jsonify({
        'captures_done': done,
        'total':         session['total'],
        'status':        session['status'],
    }), 200


@app.route('/capture-status/<session_id>', methods=['GET'])
def capture_status(session_id):
    """Poll the progress of a capture session."""
    with _capture_lock:
        session = _capture_sessions.get(session_id)
    if not session:
        return jsonify({'error': 'session not found'}), 404
    return jsonify({
        'status':          session['status'],
        'captures_done':   session['captures_done'],
        'total':           session['total'],
        'class_name':      session['class_name'],
        'training_status': session.get('training_status'),
    }), 200


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':           'ok',
        'supabase':         _db is not None,
        'patient_code':     _PATIENT_CODE or None,
        'patient_id':       _patient_id,
        'detection_running': _running_process is not None and _running_process.poll() is None,
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    host = os.environ.get('BACKEND_HOST', '0.0.0.0')
    port = int(os.environ.get('BACKEND_PORT', 5000))
    print(f'[Server] Listening on {host}:{port}')
    app.run(host=host, port=port, threaded=True)
