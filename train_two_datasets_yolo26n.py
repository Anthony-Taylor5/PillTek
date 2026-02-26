"""
train_two_datasets_yolo26n.py

You have this structure:

./v2_with_background/
  data.yaml
  train/
  valid/
  test/

./v4_with_combined/
  data.yaml
  train/
  valid/
  test/

This script can:
1) Train a YOLO26n model on ONE dataset (choose which with --dataset),
2) Optionally train on BOTH sequentially (fine-tune v4 after v2) with --dataset both,
3) Run webcam inference using the final trained weights.

Examples:
  python train_two_datasets_yolo26n.py --dataset v2 --epochs 50 --device 0
  python train_two_datasets_yolo26n.py --dataset v4 --epochs 80 --device cpu
  python train_two_datasets_yolo26n.py --dataset both --epochs 50 --epochs2 50 --device 0
 
  
    python train_two_datasets_yolo26n.py  --weights runs/detect/runs/train_v8/weights/best.pt 



Requirements:
  pip install -U ultralytics

Tip:
- If you're on Python 3.13 and things break, use Python 3.11.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO


DATASETS = {
    "v2": "v2_with_background",
    "v4": "v4_with_combined",
    "v6": "v6_with_hazards",
    "v8": "v8_with_hands"
}


def assert_dataset_layout(ds_dir: Path) -> Path:
    """
    Validate that the dataset folder contains data.yaml and train/valid/test folders.
    Returns the path to data.yaml if valid.
    """
    data_yaml = ds_dir / "data.yaml"
    missing = []

    if not data_yaml.exists():
        missing.append("data.yaml")
    for sub in ("train", "valid", "test"):
        if not (ds_dir / sub).exists():
            missing.append(sub + "/")

    if missing:
        raise FileNotFoundError(
            f"Dataset folder '{ds_dir}' is missing: {', '.join(missing)}\n"
            f"Expected structure:\n"
            f"  {ds_dir}/data.yaml\n"
            f"  {ds_dir}/train/\n"
            f"  {ds_dir}/valid/\n"
            f"  {ds_dir}/test/\n"
        )

    return data_yaml

def train_once(
    model_weights: str,
    data_yaml: Path,
    run_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project: str,
) -> Path:
    """
    Train starting from model_weights on the dataset described by data_yaml.
    Returns path to best.pt.

    Device mapping logic:
      - device == "gpu" -> resolved_device = "0"
      - device == "cpu" -> resolved_device = "cpu"
    """
   
    if device == "gpu":
        resolved_device = "0"  # use GPU id 0 for Ultralytics / PyTorch
        print("[DEVICE] Selected device: gpu (mapped to '0'). Attempting to use Radeon GPU.")
        print("[DEVICE] If this falls back to CPU, check your PyTorch / ROCm / driver installation.")
        # Helpful hint: reduce batch size or imgsz if you get out of memory errors.
    else:
        resolved_device = "cpu"
        print("[DEVICE] Selected device: cpu. Forcing CPU training.")

    model = YOLO(model_weights)

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=resolved_device,
        project=project,
        name=run_name,
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else (Path(project) / run_name)
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Training completed but best.pt not found at: {best_pt}")
    return best_pt


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLO26n on v2/v4/v6/v8 datasets and run webcam inference.")
    parser.add_argument(
        "--dataset",
        choices=["v2", "v4", "v6", "v8", "both"],
        default="v8",
        help="Which dataset to train on (v2, v4, v6, v8 or all sequentially)",
    )
    parser.add_argument("--base-model", default="yolo26n.pt", help="Starting weights (default: yolo26n.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for first training stage")
    parser.add_argument("--epochs2", type=int, default=50, help="Epochs for second stage (only if --dataset both)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="cpu", help="Device to use: gpu (Radeon 780M) or cpu. Default is gpu.")
    parser.add_argument("--project", default="runs", help="Runs output folder")
    parser.add_argument("--weights", default=None, help="Weights for inference (required if --mode infer)")
    parser.add_argument("--conf", type=float, default=0.70, help="Confidence threshold for webcam inference")
    args = parser.parse_args()

    cwd = Path.cwd()

    # Resolve dataset folders
    v2_dir = cwd / DATASETS["v2"]
    v4_dir = cwd / DATASETS["v4"]
    v6_dir = cwd / DATASETS["v6"]
    v8_dir = cwd / DATASETS["v8"]

    final_weights: Path | None = None

    if args.dataset == "v2":
        data_yaml = assert_dataset_layout(v2_dir)
        print(f"[INFO] Training on v2 dataset: {v2_dir}")
        final_weights = train_once(
            model_weights=args.base_model,
            data_yaml=data_yaml,
            run_name="train_v2",
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
        )

    elif args.dataset == "v4":
        data_yaml = assert_dataset_layout(v4_dir)
        print(f"[INFO] Training on v4 dataset: {v4_dir}")
        final_weights = train_once(
            model_weights=args.base_model,
            data_yaml=data_yaml,
            run_name="train_v4",
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
        )

    elif args.dataset == "v6":
        data_yaml = assert_dataset_layout(v6_dir)
        print(f"[INFO] Training on v6 dataset: {v6_dir}")
        final_weights = train_once(
            model_weights=args.base_model,
            data_yaml=data_yaml,
            run_name="train_v6",
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
        )

    elif args.dataset == "v8":
        data_yaml = assert_dataset_layout(v8_dir)
        print(f"[INFO] Training on v8 dataset: {v8_dir}")
        final_weights = train_once(
            model_weights=args.base_model,
            data_yaml=data_yaml,
            run_name="train_v8",
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
        )

    else:  # both
        data_yaml_v2 = assert_dataset_layout(v2_dir)
        data_yaml_v4 = assert_dataset_layout(v4_dir)
        data_yaml_v6 = assert_dataset_layout(v6_dir)
        data_yaml_v8 = assert_dataset_layout(v8_dir)

        print(f"[INFO] Stage 1: Training on v2 dataset: {v2_dir}")
        best_v2 = train_once(
            model_weights=args.base_model,
            data_yaml=data_yaml_v2,
            run_name="train_v2",
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
        )
        print(f"[INFO] Stage 1 complete. Best weights: {best_v2}")

        print(f"[INFO] Stage 2: Fine-tuning on v4 dataset: {v4_dir}")
        best_v4 = train_once(
            model_weights=str(best_v2),
            data_yaml=data_yaml_v4,
            run_name="train_v4_finetune_from_v2",
            epochs=args.epochs2,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
        )
        print(f"[INFO] Stage 2 complete. Best weights: {best_v4}")

        print(f"[INFO] Stage 3: Fine-tuning on v6 dataset: {v6_dir}")
        best_v6 = train_once(
            model_weights=str(best_v4),
            data_yaml=data_yaml_v6,
            run_name="train_v6_finetune_from_v4",
            epochs=args.epochs2,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
        )
        print(f"[INFO] Stage 3 complete. Best weights: {best_v6}")

        print(f"[INFO] Stage 4: Fine-tuning on v8 dataset: {v8_dir}")
        best_v8 = train_once(
            model_weights=str(best_v6),
            data_yaml=data_yaml_v8,
            run_name="train_v8_finetune_from_v6",
            epochs=args.epochs2,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
        )
        print(f"[INFO] Stage 4 complete. Best weights: {best_v8}")


        final_weights = best_v8

    print(f"[INFO] Final trained weights: {final_weights}")


    return 0


if __name__ == "__main__":
    raise SystemExit(main())

