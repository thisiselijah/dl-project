import argparse
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11n-pose")
    p.add_argument("--epoch", default=100)
    p.add_argument("--imgsz", type=int, default=)
    p.add_argument("--batchsz")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(model="yolo11n-pose.pt")
    model.train(data="data/data.yaml")
    metrics = model.val()


if __name__ == "__main__":
    main()
