"""
backend/supabase_sync.py — Supabase I/O for the capture/training pipeline.

This is the only Python module that talks to Supabase from the backend
capture flow. server.py and scripts/backfill_supabase.py call into it.

All public functions are fault-tolerant: they log and return a sentinel
rather than raising, except where a caller specifically wants to detect
failure (the backfill CLI exit code).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_BUCKET_IMAGES  = "bottle-images"
_BUCKET_WEIGHTS = "model-weights"


def parse_class_name(class_name: str) -> tuple[str, str]:
    """
    Split a class name like 'anthony_taylor_advil' into (user, medication).

    Rule: last underscore separates user from medication. The medication may
    contain underscores ('tylenol_500mg') and the user may too ('bob_the_builder').
    A class name with no underscore returns (class_name, '').
    """
    if "_" not in class_name:
        return class_name, ""
    user, _, med = class_name.rpartition("_")
    return user, med
