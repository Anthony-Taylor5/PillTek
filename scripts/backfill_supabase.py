"""
scripts/backfill_supabase.py — one-shot backfill of existing local data to Supabase.

Examples:
    # Single class
    python scripts/backfill_supabase.py --class-name anthony_taylor_advil

    # Single class + register an existing weights file as user_models row
    python scripts/backfill_supabase.py \\
        --class-name anthony_taylor_advil \\
        --include-weights runs/user_tuned_1/weights/best.pt \\
        --base-model v10

    # Every directory under user_bottles/ that contains an images/ subfolder
    python scripts/backfill_supabase.py --all
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as a plain script: ensure repo root is on sys.path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

# Load .env if available so SUPABASE_* vars are populated
try:
    from dotenv import load_dotenv
    load_dotenv(_REPO / ".env")
except ImportError:
    pass

from backend import supabase_sync


def _backfill_class(class_name: str, dataset_root: Path) -> int:
    dataset_dir = dataset_root / class_name
    if not (dataset_dir / "images").is_dir():
        print(f"[Backfill] skip {class_name}: no images/ subdir")
        return 0
    uploaded, failed = supabase_sync.upload_dataset_images(class_name, dataset_dir)
    print(f"[Backfill] {class_name}: {uploaded} uploaded, {failed} failed")
    return failed


def _backfill_weights(class_name: str, weights_path: Path, base_model: str) -> int:
    if not weights_path.exists():
        print(f"[Backfill] weights not found: {weights_path}", file=sys.stderr)
        return 1
    version = supabase_sync.compute_next_version(class_name)
    model_id = supabase_sync.insert_model_row(
        class_name=class_name,
        base_model=base_model,
        dataset_path=str(Path("user_bottles") / class_name),
        version=version,
        status="training",
    )
    storage_path = supabase_sync.upload_model_weights(
        class_name=class_name,
        weights_path=weights_path,
        version=version,
    )
    if storage_path and model_id:
        try:
            local_path = str(weights_path.relative_to(_REPO))
        except ValueError:
            local_path = str(weights_path)
        supabase_sync.update_model_status(
            model_id, status="ready",
            weights_local_path=local_path,
            weights_storage_path=f"model-weights/{storage_path}",
        )
        return 0
    if model_id:
        supabase_sync.update_model_status(model_id, status="failed")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill local user_bottles + weights to Supabase.")
    ap.add_argument("--class-name", help="Class folder under user_bottles/ to upload.")
    ap.add_argument("--all", action="store_true",
                    help="Process every directory under user_bottles/ with an images/ subdir.")
    ap.add_argument("--dataset-root", default="user_bottles",
                    help="Root folder containing class subfolders (default: user_bottles).")
    ap.add_argument("--include-weights", default=None,
                    help="Path to a best.pt to register under user_models for --class-name.")
    ap.add_argument("--base-model", default="v10",
                    help="Label for the base model used to fine-tune (default: v10).")
    args = ap.parse_args()

    if not args.class_name and not args.all:
        ap.error("provide --class-name or --all")
    if args.include_weights and not args.class_name:
        ap.error("--include-weights requires --class-name")

    dataset_root = (_REPO / args.dataset_root) if not Path(args.dataset_root).is_absolute() else Path(args.dataset_root)

    failures = 0
    if args.all:
        for child in sorted(dataset_root.iterdir()):
            if child.is_dir():
                failures += _backfill_class(child.name, dataset_root)
    else:
        failures += _backfill_class(args.class_name, dataset_root)

    if args.include_weights:
        weights = Path(args.include_weights)
        if not weights.is_absolute():
            weights = _REPO / weights
        failures += _backfill_weights(args.class_name, weights, args.base_model)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
