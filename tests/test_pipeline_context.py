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
