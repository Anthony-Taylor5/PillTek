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
import threading
import uuid
from datetime import datetime, timezone
from threading import Lock

from flask import Flask, request, jsonify

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


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

_running_process: subprocess.Popen | None = None

# ── Capture sessions ──────────────────────────────────────────────────────────
_capture_sessions: dict[str, dict] = {}
_capture_lock = Lock()

_ESP32_STREAM_URL = os.environ.get('ESP32_STREAM_URL', 'http://192.168.0.31:81/stream')
_YOLO_WEIGHTS     = os.environ.get('YOLO_WEIGHTS', 'runs/detect/runs/train_v10/weights/best.pt')
_REPO_ROOT        = os.path.join(os.path.dirname(__file__), '..')


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


@app.route('/start-capture', methods=['POST'])
def start_capture():
    """
    Launch capture_user_bottles.py for one medication bottle.

    Expected JSON body:
    {
      "class_name": "john_tylenol_500mg",   // sanitized medication identifier
      "source":     "http://...:81/stream", // optional — overrides ESP32_STREAM_URL
      "weights":    "runs/.../best.pt"      // optional — overrides YOLO_WEIGHTS
    }

    Returns: { "session_id": "<uuid>" }
    """
    data       = request.get_json(silent=True) or {}
    class_name = data.get('class_name', 'user_bottle')
    source     = data.get('source',  _ESP32_STREAM_URL)
    weights    = data.get('weights', _YOLO_WEIGHTS)

    session_id = str(uuid.uuid4())
    session    = {
        'session_id':    session_id,
        'class_name':    class_name,
        'status':        'running',
        'captures_done': 0,
        'total':         24,
    }
    with _capture_lock:
        _capture_sessions[session_id] = session

    cmd = [
        sys.executable, 'capture_user_bottles.py',
        '--class-name', class_name,
        '--source',     source,
        '--weights',    weights,
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=_REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        with _capture_lock:
            session['status'] = 'error'
        return jsonify({'error': str(e)}), 500

    def _read_output():
        for line in proc.stdout:
            line = line.strip()
            print(f'[Capture:{session_id[:8]}] {line}')
            if line.startswith('[CAPTURE]'):
                # e.g. "[CAPTURE] 3/24 saved."
                try:
                    n = int(line.split()[1].split('/')[0])
                    with _capture_lock:
                        session['captures_done'] = n
                except Exception:
                    pass
        proc.wait()
        with _capture_lock:
            session['status'] = 'done' if proc.returncode == 0 else 'error'

    threading.Thread(target=_read_output, daemon=True).start()

    return jsonify({'session_id': session_id}), 200


@app.route('/capture-status/<session_id>', methods=['GET'])
def capture_status(session_id):
    """
    Poll the progress of a running capture session.

    Returns:
    {
      "status":        "running" | "done" | "error",
      "captures_done": 12,
      "total":         24,
      "class_name":    "john_tylenol_500mg"
    }
    """
    with _capture_lock:
        session = _capture_sessions.get(session_id)
    if not session:
        return jsonify({'error': 'session not found'}), 404
    return jsonify({
        'status':        session['status'],
        'captures_done': session['captures_done'],
        'total':         session['total'],
        'class_name':    session['class_name'],
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
    app.run(host=host, port=port)
