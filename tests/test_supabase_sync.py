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
