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
