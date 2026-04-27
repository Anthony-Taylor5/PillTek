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
