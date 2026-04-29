"""Tests for backend.server pipeline helpers and routes."""
import json
from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def fake_db():
    """A MagicMock supabase client with chainable .table().select().eq()… ."""
    return MagicMock()


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


def test_trigger_beacon_near_passes_label_map_via_env(fake_db, monkeypatch):
    from backend import server

    server._db = fake_db
    server._patient_id = 'pid-1'   # PATIENT_CODE-resolved id is cached on the module
    server._running_process = None
    monkeypatch.setattr(server, '_ESP32_STREAM_URL', 'http://esp32/stream')
    monkeypatch.setattr(server, '_YOLO_WEIGHTS',     '/weights/best.pt')

    # medications query
    meds_chain = fake_db.table.return_value.select.return_value.eq.return_value
    meds_chain.execute.return_value = MagicMock(data=[
        {'id': 'u-A', 'name': 'Advil',   'label_code': 'A'},
        {'id': 'u-D', 'name': 'Lipitor', 'label_code': 'D'},
    ])

    captured = {}
    def fake_popen(cmd, cwd=None, env=None, **kw):
        captured['cmd'] = cmd
        captured['env'] = env
        proc = MagicMock()
        proc.poll.return_value = None
        return proc

    with patch.object(server.subprocess, 'Popen', side_effect=fake_popen):
        client = server.app.test_client()
        resp = client.post('/trigger', json={'event': 'beacon_near'})
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


def test_pipeline_debug_returns_resolved_context(fake_db):
    """Simulates the full lookup path without spawning a subprocess."""
    from backend import server
    server._db = fake_db
    server._patient_id = 'pid-1'   # PATIENT_CODE already resolved
    meds_chain = fake_db.table.return_value.select.return_value.eq.return_value
    meds_chain.execute.return_value = MagicMock(data=[
        {'id': 'u-A', 'name': 'Advil', 'label_code': 'A'},
    ])
    client = server.app.test_client()
    resp = client.post('/pipeline-debug', json={})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['patient_id'] == 'pid-1'
    assert body['label_map']  == {'A': 'u-A'}
    assert body['env_preview']['PILLTEK_LABEL_MAP'] == '{"A": "u-A"}'
