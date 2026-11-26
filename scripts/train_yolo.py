"""Train a YOLO-Pose model on the MTDDH hip keypoint dataset.

Examples:
    # quick smoke test
    python scripts/train_yolo.py --epochs 1 --fraction 0.1

    # full run on Colab, saving results to Google Drive
    python scripts/train_yolo.py --data /content/mtddh/data.yaml \
        --project /content/drive/MyDrive/dl-project/runs --name y11n_640

    # resume an interrupted run
    python scripts/train_yolo.py --resume /content/drive/MyDrive/dl-project/runs/y11n_640/weights/last.pt
"""

import argparse

import torch
import ultralytics
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO-Pose on MTDDH")
    # model / data
    p.add_argument("--model", default="yolo11n-pose.pt", help="model name or .pt path")
    p.add_argument("--data", default="data/data.yaml", help="path to data.yaml")
    p.add_argument("--device", default=None, help="0 / mps / cpu (default: auto)")
    # hyperparameters
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--optimizer", default="auto", help="SGD / Adam / AdamW / auto")
    p.add_argument("--lr0", type=float, default=0.01, help="initial LR (ignored when optimizer=auto)")
    p.add_argument("--patience", type=int, default=100, help="early-stopping patience in epochs")
    p.add_argument("--seed", type=int, default=0)
    # runtime
    p.add_argument("--cache", action="store_true", help="cache images in RAM")
    p.add_argument("--fraction", type=float, default=1.0, help="fraction of train set to use")
    p.add_argument("--project", default="runs/pose", help="output root (point to Drive on Colab)")
    p.add_argument("--name", default=None, help="run name under --project")
    p.add_argument("--resume", default=None, help="path to last.pt to resume from")
    return p.parse_args()


def pick_device(device):
    if device is not None:
        return device
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()

    if args.resume:
        # all settings are restored from the checkpoint's saved args
        model = YOLO(args.resume)
        model.train(resume=True)
    else:
        model_path = args.model if args.model.endswith(".pt") else args.model + ".pt"
        model = YOLO(model_path)
        model.train(
            data=args.data,
            device=pick_device(args.device),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            optimizer=args.optimizer,
            lr0=args.lr0,
            patience=args.patience,
            seed=args.seed,
            cache=args.cache,
            fraction=args.fraction,
            project=args.project,
            name=args.name,
        )

    print(f"best weights: {model.trainer.best}")
    print(f"run dir:      {model.trainer.save_dir}")


if __name__ == "__main__":
    ultralytics.checks()
    main()
