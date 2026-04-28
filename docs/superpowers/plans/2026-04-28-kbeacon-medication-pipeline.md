# KBeacon → Medication Detection Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the ESP32 KBeacon scanner reports `beacon_near`, the backend resolves which patient owns that beacon, fetches that patient's medications filtered to the allowed label set `{A, B, D, F}`, and hands a `label_code → medication_id` map to the YOLO detection subprocess. The subprocess uses that map to populate `CLASS_TO_MED_ID` from `model.names` at startup, so when a hand-bottle overlap fires, the corresponding medication row is upserted to `medication_logs.status='Taken'` for the correct medication and patient.

**Architecture:** The transport already works: `cam_and_server_code.ino → POST /trigger → subprocess.Popen(test_with_hand_recognition.py) → POST /detection-event → medication_logs upsert`. What's missing is the *context* travelling along that path. We add it as: (a) a new `medications.label_code` column, (b) two backend helpers (patient-by-beacon, allowed-meds), (c) a pure mapping function in a new module, (d) `/trigger` enrichment to pass a JSON env var into the subprocess, (e) startup hydration of `CLASS_TO_MED_ID` in `test_with_hand_recognition.py`, (f) a `/pipeline-debug` endpoint that exercises the full flow without hardware. All Supabase access is mocked in tests; the existing `tests/test_capture_headless.py` style (`unittest.mock.patch`) is the model.

**Tech Stack:** Python 3, Flask, `supabase-py`, Ultralytics YOLO, MediaPipe, OpenCV, pytest + `unittest.mock`.

---

## File Structure

**Modify:**
- `backend/server.py` — new helpers, enriched `/trigger`, new `/pipeline-debug` endpoint.
- `test_with_hand_recognition.py` — hydrate `CLASS_TO_MED_ID` from `PILLTEK_LABEL_MAP` env var on startup.
- `supabase/schema.sql` — add `label_code` column to `medications`.

**Create:**
- `backend/pipeline_context.py` — pure functions: `build_label_med_map`, `model_class_to_med_id`. No I/O, fully unit-testable.
- `tests/test_pipeline_context.py` — unit tests for `pipeline_context` (pure).
- `tests/test_server_trigger.py` — tests for `/trigger`, `/pipeline-debug`, helpers (mocked Supabase + mocked Popen).
- `tests/test_detection_label_map.py` — tests for `CLASS_TO_MED_ID` hydration in `test_with_hand_recognition.py`.

**Each step is one action. Code blocks are complete — no placeholders.**

---

## Task 1: Add `label_code` column to medications schema

**Files:**
- Modify: `supabase/schema.sql` (append migration block at end of file)

- [ ] **Step 1: Append the additive migration**

Add this block at the end of `supabase/schema.sql`:

```sql
-- ── 2026-04-28: medications.label_code (KBeacon pipeline) ────────────────────
-- Optional letter code used by the YOLO model class names ("Bottle A", etc.).
-- Allowed values match the trained model's bottle classes; nullable so
-- existing rows are unaffected.
ALTER TABLE medications ADD COLUMN IF NOT EXISTS label_code text
  CHECK (label_code IS NULL OR label_code IN ('A','B','D','F'));
CREATE INDEX IF NOT EXISTS idx_medications_patient_label
  ON medications (patient_id, label_code);
```

- [ ] **Step 2: Verify SQL parses**

Run: `python -c "open('supabase/schema.sql').read()"`
Expected: no error (the file just needs to be valid text; we cannot execute against Supabase from CI).

- [ ] **Step 3: Commit**

```bash
git add supabase/schema.sql
git commit -m "Add label_code column to medications for KBeacon pipeline"
```

---

## Task 2: Pure module `backend/pipeline_context.py`

**Files:**
- Create: `backend/pipeline_context.py`
- Test: `tests/test_pipeline_context.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline_context.py`:

```python
"""Unit tests for backend.pipeline_context — pure, no I/O."""
import pytest
from backend.pipeline_context import (
    ALLOWED_LABELS,
    build_label_med_map,
    model_class_to_med_id,
)


def test_allowed_labels_is_the_required_set():
    assert ALLOWED_LABELS == frozenset({'A', 'B', 'D', 'F'})


def test_build_label_med_map_keeps_allowed_only():
    meds = [
        {'id': 'u-A', 'name': 'Advil',   'label_code': 'A'},
        {'id': 'u-B', 'name': 'Tylenol', 'label_code': 'B'},
        {'id': 'u-C', 'name': 'Aspirin', 'label_code': 'C'},   # not allowed
        {'id': 'u-N', 'name': 'NoLabel', 'label_code': None},
        {'id': 'u-D', 'name': 'Lipitor', 'label_code': 'd'},   # case-insensitive
    ]
    assert build_label_med_map(meds) == {'A': 'u-A', 'B': 'u-B', 'D': 'u-D'}


def test_build_label_med_map_handles_empty():
    assert build_label_med_map([]) == {}


def test_model_class_to_med_id_extracts_trailing_letter():
    # YOLO model.names is typically a dict {class_id: name}
    names = {0: '-', 1: 'Bottle A', 2: 'Bottle B', 3: 'Bottle D', 4: 'Bottle F'}
    label_map = {'A': 'u-A', 'B': 'u-B', 'D': 'u-D', 'F': 'u-F'}
    assert model_class_to_med_id(names, label_map) == {
        1: 'u-A', 2: 'u-B', 3: 'u-D', 4: 'u-F',
    }


def test_model_class_to_med_id_skips_unmapped_and_non_letter():
    names = {0: '-', 1: 'Bottle A', 2: 'Hazard', 3: 'Bottle Z'}
    label_map = {'A': 'u-A'}
    assert model_class_to_med_id(names, label_map) == {1: 'u-A'}


def test_model_class_to_med_id_accepts_list_form():
    # Some YOLO versions expose names as a list
    names = ['-', 'Bottle A', 'Bottle B']
    label_map = {'A': 'u-A', 'B': 'u-B'}
    assert model_class_to_med_id(names, label_map) == {1: 'u-A', 2: 'u-B'}
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_pipeline_context.py -v`
Expected: ImportError / ModuleNotFoundError on `backend.pipeline_context`.

- [ ] **Step 3: Implement the module**

Create `backend/pipeline_context.py`:

```python
"""Pure helpers for the KBeacon → detection pipeline. No I/O, no Supabase.

Tested by tests/test_pipeline_context.py.
"""
from __future__ import annotations

import re
from typing import Iterable, Mapping

ALLOWED_LABELS: frozenset[str] = frozenset({'A', 'B', 'D', 'F'})

# Pull the trailing single-letter token from a YOLO class name,
# e.g. "Bottle A" → "A", "bottle  d" → "D", "Hazard" → None.
_TRAILING_LETTER = re.compile(r'\b([A-Za-z])\s*$')


def build_label_med_map(medications: Iterable[Mapping]) -> dict[str, str]:
    """Filter medications to ALLOWED_LABELS and return {label: medication_id}.

    Last-write wins on duplicate label codes (medications table allows it),
    which is intentional: the most recently added medication for a label is
    the one bound to that bottle.
    """
    out: dict[str, str] = {}
    for m in medications:
        code = (m.get('label_code') or '').strip().upper()
        if code in ALLOWED_LABELS:
            out[code] = m['id']
    return out


def model_class_to_med_id(
    model_names: Mapping[int, str] | list[str],
    label_med_map: Mapping[str, str],
) -> dict[int, str]:
    """Convert YOLO class id → medication_id using a label-letter map.

    Accepts either dict-form or list-form `model.names`.  Class names whose
    trailing letter is not in `label_med_map` are skipped silently.
    """
    if isinstance(model_names, list):
        items = enumerate(model_names)
    else:
        items = model_names.items()

    out: dict[int, str] = {}
    for class_id, name in items:
        m = _TRAILING_LETTER.search(name or '')
        if not m:
            continue
        code = m.group(1).upper()
        med = label_med_map.get(code)
        if med:
            out[int(class_id)] = med
    return out
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/test_pipeline_context.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline_context.py tests/test_pipeline_context.py
git commit -m "Add pipeline_context module for label-to-medication mapping"
```

---

## Task 3: Backend helper `_resolve_patient_by_beacon`

**Files:**
- Modify: `backend/server.py` (insert helper near `_resolve_patient_id` at line 72)
- Test: `tests/test_server_trigger.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_server_trigger.py`:

```python
"""Tests for backend.server pipeline helpers and /trigger route."""
import json
from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def fake_db():
    """A MagicMock supabase client with chainable .table().select().eq()… ."""
    db = MagicMock()
    return db


def _stub_patients_query(db, mac, patient_row):
    """Wire db.table('patients').select('id,caregiver_uid')
            .eq('beacon_mac', mac).maybe_single().execute() → patient_row."""
    table   = db.table.return_value
    select  = table.select.return_value
    eq      = select.eq.return_value
    single  = eq.maybe_single.return_value
    single.execute.return_value = MagicMock(data=patient_row)


def test_resolve_patient_by_beacon_uses_mac_when_present(fake_db):
    from backend import server
    server._db = fake_db
    server._patient_id = None
    _stub_patients_query(
        fake_db, 'aa:bb:cc:dd:ee:ff',
        {'id': 'pid-mac', 'caregiver_uid': 'fb-uid-X'},
    )
    pid, caregiver = server._resolve_patient_by_beacon('AA:BB:CC:DD:EE:FF')
    assert pid == 'pid-mac'
    assert caregiver == 'fb-uid-X'
    fake_db.table.assert_called_with('patients')
    fake_db.table.return_value.select.return_value.eq.assert_called_with(
        'beacon_mac', 'aa:bb:cc:dd:ee:ff',
    )


def test_resolve_patient_by_beacon_falls_back_to_patient_code(fake_db):
    """If MAC lookup returns nothing, fall back to PATIENT_CODE env."""
    from backend import server
    server._db = fake_db
    server._patient_id = None
    server._PATIENT_CODE = 'PTK-FAKE'
    _stub_patients_query(fake_db, 'aa:bb', None)

    # second call — the code-based resolver
    code_table  = MagicMock()
    code_select = code_table.select.return_value
    code_eq     = code_select.eq.return_value
    code_single = code_eq.maybe_single.return_value
    code_single.execute.return_value = MagicMock(data={'id': 'pid-code'})
    fake_db.table.side_effect = [
        # first call (mac lookup) returns the original chain
        fake_db.table.return_value,
        code_table,
    ]
    pid, caregiver = server._resolve_patient_by_beacon('aa:bb')
    assert pid == 'pid-code'
    assert caregiver is None


def test_resolve_patient_by_beacon_returns_none_on_no_db(fake_db):
    from backend import server
    server._db = None
    pid, caregiver = server._resolve_patient_by_beacon('aa:bb')
    assert (pid, caregiver) == (None, None)
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `pytest tests/test_server_trigger.py::test_resolve_patient_by_beacon_uses_mac_when_present -v`
Expected: AttributeError on `server._resolve_patient_by_beacon`.

- [ ] **Step 3: Implement helper in `backend/server.py`**

Insert after the existing `_resolve_patient_id` function (after line 86):

```python
def _resolve_patient_by_beacon(beacon_mac: str | None) -> tuple[str | None, str | None]:
    """Resolve (patient_id, caregiver_uid) by BLE MAC address.

    Falls back to PATIENT_CODE env-based lookup if mac is None or unmapped.
    Returns (None, None) if Supabase is not configured.
    """
    if not _db:
        return (None, None)
    if beacon_mac:
        try:
            res = (
                _db.table('patients')
                   .select('id,caregiver_uid')
                   .eq('beacon_mac', beacon_mac.lower())
                   .maybe_single()
                   .execute()
            )
            if res and res.data:
                print(f'[Pipeline] beacon→patient: {beacon_mac} → {res.data["id"]}')
                return (res.data['id'], res.data.get('caregiver_uid'))
        except Exception as e:
            print(f'[Pipeline] beacon→patient lookup failed: {e}')

    # Fallback: PATIENT_CODE env var (existing behavior)
    return (_resolve_patient_id(), None)
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/test_server_trigger.py -v -k resolve_patient`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/server.py tests/test_server_trigger.py
git commit -m "Resolve patient by beacon MAC with PATIENT_CODE fallback"
```

---

## Task 4: Backend helper `_fetch_allowed_medications`

**Files:**
- Modify: `backend/server.py` (insert after `_resolve_patient_by_beacon`)
- Modify: `tests/test_server_trigger.py` (append tests)

- [ ] **Step 1: Append the failing test**

Add to `tests/test_server_trigger.py`:

```python
def test_fetch_allowed_medications_filters_to_allowed_set(fake_db):
    from backend import server
    server._db = fake_db
    rows = [
        {'id': 'u-A', 'name': 'Advil',   'label_code': 'A'},
        {'id': 'u-X', 'name': 'NoLabel', 'label_code': None},
        {'id': 'u-D', 'name': 'Lipitor', 'label_code': 'D'},
        {'id': 'u-C', 'name': 'Aspirin', 'label_code': 'C'},
    ]
    chain = fake_db.table.return_value.select.return_value.eq.return_value
    chain.execute.return_value = MagicMock(data=rows)
    out = server._fetch_allowed_medications('pid-1')
    assert out == {'A': 'u-A', 'D': 'u-D'}
    fake_db.table.assert_called_with('medications')


def test_fetch_allowed_medications_empty_on_no_patient(fake_db):
    from backend import server
    server._db = fake_db
    assert server._fetch_allowed_medications(None) == {}
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_server_trigger.py -v -k fetch_allowed`
Expected: AttributeError on `server._fetch_allowed_medications`.

- [ ] **Step 3: Implement the helper**

Add to `backend/server.py` after `_resolve_patient_by_beacon`. Also add the import at the top of the file (just below `import requests as _requests`):

```python
from backend.pipeline_context import build_label_med_map  # may need sys.path fixup; see import block below
```

If the existing import block already routes `from backend import supabase_sync` with a sys.path fallback, mirror that same pattern for `pipeline_context`:

```python
try:
    from backend.pipeline_context import build_label_med_map
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from pipeline_context import build_label_med_map  # type: ignore[no-redef]
```

Then the helper:

```python
def _fetch_allowed_medications(patient_id: str | None) -> dict[str, str]:
    """Return {label_code: medication_id} for the patient's allowed-label meds."""
    if not _db or not patient_id:
        return {}
    try:
        res = (
            _db.table('medications')
               .select('id,name,label_code')
               .eq('patient_id', patient_id)
               .execute()
        )
        meds = res.data or []
        out  = build_label_med_map(meds)
        print(f'[Pipeline] patient {patient_id}: label_map={out}')
        return out
    except Exception as e:
        print(f'[Pipeline] _fetch_allowed_medications failed: {e}')
        return {}
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/test_server_trigger.py -v -k fetch_allowed`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/server.py tests/test_server_trigger.py
git commit -m "Fetch allowed-label medications from Supabase by patient"
```

---

## Task 5: Enrich `/trigger` to pass medication context to detection

**Files:**
- Modify: `backend/server.py` lines 229–289 (the `/trigger` route)
- Modify: `tests/test_server_trigger.py` (append /trigger tests)

- [ ] **Step 1: Append the failing test**

Add to `tests/test_server_trigger.py`:

```python
def test_trigger_beacon_near_passes_label_map_via_env(fake_db, monkeypatch):
    from backend import server

    server._db = fake_db
    server._patient_id = None
    server._running_process = None
    monkeypatch.setattr(server, '_ESP32_STREAM_URL', 'http://esp32/stream')
    monkeypatch.setattr(server, '_YOLO_WEIGHTS',     '/weights/best.pt')

    # patient lookup
    _stub_patients_query(
        fake_db, 'aa:bb:cc:dd:ee:ff',
        {'id': 'pid-1', 'caregiver_uid': 'fb-X'},
    )
    # medications query (second .table call)
    meds_table = MagicMock()
    meds_chain = meds_table.select.return_value.eq.return_value
    meds_chain.execute.return_value = MagicMock(data=[
        {'id': 'u-A', 'name': 'Advil',   'label_code': 'A'},
        {'id': 'u-D', 'name': 'Lipitor', 'label_code': 'D'},
    ])
    fake_db.table.side_effect = [
        fake_db.table.return_value,   # patients lookup chain stays as-is
        meds_table,
    ]

    captured = {}
    def fake_popen(cmd, cwd=None, env=None, **kw):
        captured['cmd'] = cmd
        captured['env'] = env
        proc = MagicMock()
        proc.poll.return_value = None
        return proc

    with patch.object(server.subprocess, 'Popen', side_effect=fake_popen):
        client = server.app.test_client()
        resp = client.post('/trigger', json={
            'event':      'beacon_near',
            'beacon_mac': 'aa:bb:cc:dd:ee:ff',
        })
    assert resp.status_code == 200
    assert '--source'  in captured['cmd']
    assert '--weights' in captured['cmd']
    label_map = json.loads(captured['env']['PILLTEK_LABEL_MAP'])
    assert label_map == {'A': 'u-A', 'D': 'u-D'}
    assert captured['env']['PILLTEK_PATIENT_ID'] == 'pid-1'


def test_trigger_invalid_event_returns_400(fake_db):
    from backend import server
    client = server.app.test_client()
    resp = client.post('/trigger', json={'event': 'nonsense'})
    assert resp.status_code == 400
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_server_trigger.py -v -k trigger`
Expected: AssertionError — current `/trigger` does not set `PILLTEK_LABEL_MAP`.

- [ ] **Step 3: Replace the `/trigger` route**

In `backend/server.py`, replace the entire `/trigger` route (currently at lines 229–289 after the Task 0 merge edit):

```python
@app.route('/trigger', methods=['POST'])
def trigger():
    """ESP32 beacon webhook. Spawns / terminates the detection subprocess
    and logs each event to Supabase, with per-patient medication context.

    Expected JSON body (all fields optional):
      { "event": "beacon_near" | "beacon_far",
        "beacon_mac": "dd:34:02:0a:2d:f1" }
    """
    global _running_process

    data       = request.get_json(silent=True) or {}
    event      = data.get('event', '')
    beacon_mac = (data.get('beacon_mac') or '').lower() or None
    print(f'[ESP32] event={event} beacon_mac={beacon_mac}')

    if event == 'beacon_near':
        patient_id, caregiver_uid = _resolve_patient_by_beacon(beacon_mac)
        label_map = _fetch_allowed_medications(patient_id)
        _log_event('beacon_near', raw_meta={
            'source':         'esp32',
            'beacon_mac':     beacon_mac,
            'patient_id':     patient_id,
            'label_map_keys': sorted(label_map.keys()),
        })

        if _running_process is None or _running_process.poll() is not None:
            cmd = [
                sys.executable, 'test_with_hand_recognition.py',
                '--source',  _ESP32_STREAM_URL,
                '--weights', _YOLO_WEIGHTS,
            ]
            child_env = os.environ.copy()
            child_env.setdefault(
                'PILLTEK_BACKEND',
                f"http://127.0.0.1:{os.environ.get('BACKEND_PORT', '5000')}",
            )
            child_env['PILLTEK_LABEL_MAP'] = json.dumps(label_map)
            if patient_id:
                child_env['PILLTEK_PATIENT_ID'] = patient_id
            if caregiver_uid:
                child_env['PILLTEK_CAREGIVER_UID'] = caregiver_uid
            print(
                f'[Pipeline] starting detection: patient={patient_id} '
                f'labels={sorted(label_map)} cmd={" ".join(cmd)}'
            )
            _running_process = subprocess.Popen(cmd, cwd=_REPO_ROOT, env=child_env)
        else:
            print('[Server] Detection subprocess already running.')
        return 'ok'

    elif event == 'beacon_far':
        _log_event('beacon_far', raw_meta={'source': 'esp32', 'beacon_mac': beacon_mac})
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

    return jsonify({'error': 'invalid event'}), 400
```

Add `import json` to the imports near the top of `backend/server.py` if it isn't already there.

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/test_server_trigger.py -v`
Expected: all `test_trigger_*` and earlier helper tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/server.py tests/test_server_trigger.py
git commit -m "Pass per-patient medication label map to detection via env"
```

---

## Task 6: Hydrate `CLASS_TO_MED_ID` in `test_with_hand_recognition.py`

**Files:**
- Modify: `test_with_hand_recognition.py` (top of file, before `run_infer`)
- Test: `tests/test_detection_label_map.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_detection_label_map.py`:

```python
"""Tests CLASS_TO_MED_ID hydration from PILLTEK_LABEL_MAP env var."""
import json
import os
from unittest.mock import patch
import importlib


def _reload_thwr(env):
    with patch.dict(os.environ, env, clear=False):
        import test_with_hand_recognition as thwr
        importlib.reload(thwr)
        return thwr


def test_hydrate_from_label_map_with_dict_names():
    env = {'PILLTEK_LABEL_MAP': json.dumps({'A': 'u-A', 'B': 'u-B'})}
    thwr = _reload_thwr(env)
    names = {0: '-', 1: 'Bottle A', 2: 'Bottle B', 3: 'Bottle C'}
    out = thwr.hydrate_class_to_med_id(names)
    assert out == {1: 'u-A', 2: 'u-B'}


def test_hydrate_from_label_map_with_list_names():
    env = {'PILLTEK_LABEL_MAP': json.dumps({'D': 'u-D'})}
    thwr = _reload_thwr(env)
    names = ['-', 'Bottle A', 'Bottle B', 'Bottle D']
    out = thwr.hydrate_class_to_med_id(names)
    assert out == {3: 'u-D'}


def test_hydrate_returns_empty_when_env_unset():
    # Explicitly clear; importlib.reload picks up the cleared env
    env_pop = dict(os.environ)
    env_pop.pop('PILLTEK_LABEL_MAP', None)
    with patch.dict(os.environ, env_pop, clear=True):
        import test_with_hand_recognition as thwr
        importlib.reload(thwr)
        assert thwr.hydrate_class_to_med_id({1: 'Bottle A'}) == {}
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_detection_label_map.py -v`
Expected: AttributeError on `thwr.hydrate_class_to_med_id`.

- [ ] **Step 3: Add hydration to `test_with_hand_recognition.py`**

Replace the existing `CLASS_TO_MED_ID` declaration block (currently lines 26–32 of the file). Find:

```python
# Maps YOLO class id → medication UUID in Supabase.
# Populate this with real values from your 'medications' table once meds are created.
# Example: CLASS_TO_MED_ID = { 1: "uuid-for-bottle-a", 2: "uuid-for-bottle-b" }
CLASS_TO_MED_ID: dict[int, str] = {}

_CLASS_NAMES_GLOBAL: list[str] = []          # set in run_infer after model load
_last_event_time:    dict[int, float] = {}   # per-class debounce tracker
```

Replace with:

```python
# Hydrated at runtime from the PILLTEK_LABEL_MAP env var (set by backend /trigger).
# Format of env var: JSON object {"A": "<med-uuid>", "B": "<med-uuid>", ...}.
# Fully empty by default — detection still runs but never marks medications taken.
import json as _json
import re  as _re

CLASS_TO_MED_ID: dict[int, str] = {}
_LABEL_MED_MAP: dict[str, str]  = _json.loads(os.environ.get('PILLTEK_LABEL_MAP', '{}'))
_PATIENT_ID:    str | None       = os.environ.get('PILLTEK_PATIENT_ID') or None

_CLASS_NAMES_GLOBAL: list[str] = []          # set in run_infer after model load
_last_event_time:    dict[int, float] = {}   # per-class debounce tracker

_TRAILING_LETTER = _re.compile(r'\b([A-Za-z])\s*$')


def hydrate_class_to_med_id(model_names) -> dict[int, str]:
    """Return {class_id: medication_id} from model.names + _LABEL_MED_MAP.

    Accepts model.names as either a dict {id: name} or a list [name, ...].
    Pure helper; populates the module-global CLASS_TO_MED_ID as a side
    effect for callers that already read it.
    """
    items = enumerate(model_names) if isinstance(model_names, list) else model_names.items()
    out: dict[int, str] = {}
    for class_id, name in items:
        m = _TRAILING_LETTER.search(name or '')
        if not m:
            continue
        med = _LABEL_MED_MAP.get(m.group(1).upper())
        if med:
            out[int(class_id)] = med
    CLASS_TO_MED_ID.clear()
    CLASS_TO_MED_ID.update(out)
    if out:
        print(f'[Pipeline] CLASS_TO_MED_ID hydrated: {out}')
    else:
        print('[Pipeline] CLASS_TO_MED_ID empty — no allowed-label classes found')
    return out
```

- [ ] **Step 4: Call `hydrate_class_to_med_id` after the YOLO model loads**

Find `run_infer` in `test_with_hand_recognition.py`. Immediately after the line that loads the model and assigns `_CLASS_NAMES_GLOBAL`, add the hydration call. The exact insertion is at the line just after `_CLASS_NAMES_GLOBAL = ...` is set (typically near `model = YOLO(args.weights)`):

```python
    # Existing:
    model = YOLO(args.weights)
    global _CLASS_NAMES_GLOBAL
    _CLASS_NAMES_GLOBAL = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)

    # NEW: hydrate label map from env so detection events include medication_id
    hydrate_class_to_med_id(model.names)
```

If the existing `run_infer` does not currently set `_CLASS_NAMES_GLOBAL`, add that line too.

Also update the `payload` block in `post_detection_event` so it always includes `patient_id` if known. Find the existing `payload = { ... }` dict and ensure it ends with:

```python
    if _PATIENT_ID:
        payload['patient_id'] = _PATIENT_ID
```

- [ ] **Step 5: Run tests to confirm they pass**

Run: `pytest tests/test_detection_label_map.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add test_with_hand_recognition.py tests/test_detection_label_map.py
git commit -m "Hydrate CLASS_TO_MED_ID from PILLTEK_LABEL_MAP env var"
```

---

## Task 7: `/pipeline-debug` endpoint (no hardware)

**Files:**
- Modify: `backend/server.py` (add new route near `/health`)
- Modify: `tests/test_server_trigger.py` (append tests)

- [ ] **Step 1: Append the failing test**

Add to `tests/test_server_trigger.py`:

```python
def test_pipeline_debug_returns_resolved_context(fake_db):
    """Simulates the full lookup path without spawning a subprocess."""
    from backend import server
    server._db = fake_db
    server._patient_id = None
    _stub_patients_query(
        fake_db, 'aa:bb',
        {'id': 'pid-1', 'caregiver_uid': 'fb-X'},
    )
    meds_table = MagicMock()
    meds_chain = meds_table.select.return_value.eq.return_value
    meds_chain.execute.return_value = MagicMock(data=[
        {'id': 'u-A', 'name': 'Advil', 'label_code': 'A'},
    ])
    fake_db.table.side_effect = [fake_db.table.return_value, meds_table]
    client = server.app.test_client()
    resp = client.post('/pipeline-debug', json={'beacon_mac': 'aa:bb'})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['patient_id']    == 'pid-1'
    assert body['caregiver_uid'] == 'fb-X'
    assert body['label_map']     == {'A': 'u-A'}
    assert body['env_preview']['PILLTEK_LABEL_MAP'] == '{"A": "u-A"}'
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `pytest tests/test_server_trigger.py -v -k pipeline_debug`
Expected: 404 (route does not exist).

- [ ] **Step 3: Implement the route**

Add to `backend/server.py` just before the `/health` route:

```python
@app.route('/pipeline-debug', methods=['POST'])
def pipeline_debug():
    """Diagnostic: run the patient + label map lookup that /trigger would,
    without spawning the detection subprocess. Useful for verifying that
    Supabase data is shaped correctly for the trigger flow."""
    data       = request.get_json(silent=True) or {}
    beacon_mac = (data.get('beacon_mac') or '').lower() or None
    patient_id, caregiver_uid = _resolve_patient_by_beacon(beacon_mac)
    label_map  = _fetch_allowed_medications(patient_id)
    env_preview = {
        'PILLTEK_LABEL_MAP':    json.dumps(label_map),
        'PILLTEK_PATIENT_ID':   patient_id or '',
        'PILLTEK_CAREGIVER_UID': caregiver_uid or '',
    }
    return jsonify({
        'beacon_mac':    beacon_mac,
        'patient_id':    patient_id,
        'caregiver_uid': caregiver_uid,
        'label_map':     label_map,
        'env_preview':   env_preview,
    }), 200
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/test_server_trigger.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add backend/server.py tests/test_server_trigger.py
git commit -m "Add /pipeline-debug endpoint for hardware-free trigger verification"
```

---

## Task 8: Run the full test suite and integration smoke

**Files:**
- (none modified)

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all green; only the previously `@pytest.mark.skip`-decorated test in `test_capture_headless.py` is skipped.

- [ ] **Step 2: Smoke-test `/pipeline-debug` against a live Supabase**

With `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` exported and a row in `patients` whose `beacon_mac` matches the test value, run:

```bash
python backend/server.py &
sleep 2
curl -s -X POST http://127.0.0.1:5000/pipeline-debug \
  -H 'Content-Type: application/json' \
  -d '{"beacon_mac":"dd:34:02:0a:2d:f1"}' | python -m json.tool
kill %1
```

Expected: JSON with `patient_id` populated and `label_map` containing the patient's medications whose `label_code` ∈ `{A,B,D,F}`. Empty objects are also acceptable when no rows match — the endpoint must still return 200 with the empty shapes.

- [ ] **Step 3: Smoke-test `/trigger` end-to-end (optional, requires ESP32)**

With ESP32 powered and beacon nearby, watch the backend logs:

```bash
python backend/server.py
```

Expected stdout:

```
[ESP32] event=beacon_near beacon_mac=dd:34:02:0a:2d:f1
[Pipeline] beacon→patient: dd:34:02:0a:2d:f1 → <uuid>
[Pipeline] patient <uuid>: label_map={'A': '<u-A>', ...}
[Pipeline] starting detection: patient=<uuid> labels=['A', ...] cmd=...
[Pipeline] CLASS_TO_MED_ID hydrated: {1: '<u-A>', ...}
```

Followed (when a hand grasps a bottle) by:

```
[Detection] hand_bottle_overlap | class=1 (Bottle A) conf=0.87
[Supabase] medication_logs upsert: status=Taken
```

- [ ] **Step 4: Commit any final adjustments**

If tweaks were needed for the smoke test:

```bash
git add -A
git commit -m "Tweak pipeline logging based on smoke test feedback"
```

---

## Self-Review Notes

Verified:
- Every task references concrete file paths and line numbers.
- All TDD steps include full code, no placeholders.
- The `label_code` filter in Task 4 matches the constraint added in Task 1 (`A/B/D/F`).
- `build_label_med_map` (Task 2) is the only place doing label filtering; Task 4 calls it; the names are consistent across all tasks.
- `PILLTEK_LABEL_MAP` is the env-var contract between `/trigger` (Task 5) and detection startup (Task 6) — both use the same key.
- The trailing-letter regex in `pipeline_context.py` and `test_with_hand_recognition.py` is identical, so model-name parsing is consistent on both sides.
- `medication_logs` upsert path was already wired in `backend/server.py:308–313` and is left untouched; the new context only reaches it via the hydrated `medication_id` field on the detection event.

Risks / assumptions:
- Trained YOLO model class names follow the convention "Bottle A" / "Bottle B" / "Bottle D" / "Bottle F" (i.e., the trailing token is the label letter). If the active model uses different names, `hydrate_class_to_med_id` will return `{}` and detection runs without medication tagging. To verify before training a new model, run `python -c "from ultralytics import YOLO; print(YOLO('runs/detect/runs/train_v10/weights/best.pt').names)"`.
- `medications.label_code` is operator-set. UI/CLI to set it is *not* part of this plan; for now it must be set with a SQL update in Supabase Studio.
- Caregiver notification is not in scope. Caregivers see updates via the existing app-side query of `medication_logs` (real-time subscription is a future task).
