
from __future__ import annotations
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path('.').resolve()

YOLO_DATASET_DIR = PROJECT_ROOT / 'yolo_dataset/'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'task3'

MODEL_NAME = 'yolov8n-seg.pt'
SEED = 42
IMG_SIZE = 384

assert YOLO_DATASET_DIR.exists()

model = YOLO(OUTPUT_DIR / "runs/yolo_seg_res/weights/last.pt")
model.train(resume=True)
