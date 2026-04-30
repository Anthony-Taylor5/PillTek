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
        code = (m.get('label') or '').strip().upper()
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
