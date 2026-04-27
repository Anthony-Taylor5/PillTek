"""Tests for capture_bottles.class_run_dir()."""
from pathlib import Path

from capture_bottles import class_run_dir


def test_first_run_uses_bare_class_name(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    assert class_run_dir(runs, "anthony_taylor_advil") == runs / "anthony_taylor_advil"


def test_second_run_appends_v2(tmp_path):
    runs = tmp_path / "runs"
    (runs / "anthony_taylor_advil").mkdir(parents=True)
    assert class_run_dir(runs, "anthony_taylor_advil") == runs / "anthony_taylor_advil_v2"


def test_skips_to_next_available_version(tmp_path):
    runs = tmp_path / "runs"
    (runs / "anthony_taylor_advil").mkdir(parents=True)
    (runs / "anthony_taylor_advil_v2").mkdir()
    (runs / "anthony_taylor_advil_v3").mkdir()
    assert class_run_dir(runs, "anthony_taylor_advil") == runs / "anthony_taylor_advil_v4"


def test_unrelated_class_unaffected(tmp_path):
    runs = tmp_path / "runs"
    (runs / "anthony_taylor_advil").mkdir(parents=True)
    assert class_run_dir(runs, "bob_tylenol") == runs / "bob_tylenol"
