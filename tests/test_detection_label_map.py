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
