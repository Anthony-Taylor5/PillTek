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
