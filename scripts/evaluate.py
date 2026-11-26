import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from ultralytics import YOLO
from dl_project.metrics import NME, PCK


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights/best.pt")
    p.add_argument(
        "--data",
    )
    p.add_argument(
        "--split",
    )
    p.add_argument(
        "--conf",
        type=int,
    )
    p.add_argument(
        "--imgsz",
        type=int,
    )
    return p.parse_args()


def load_gt():
    pass
  
def match():
    pass


def main():
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(
        data="data/data.yaml",
        imgsz=args.imgsz,
        batch=args.batch,
        conf=0.001,
        iou=0.5,
        device=0,
    )

    results = [{"mAP50-95": metrics.box.map}]
    df = pd.DataFrame(results)

    with open("sweep_log.md", "a", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False) + "\n")


if __name__ == "__main__":
    main()
