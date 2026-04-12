
from __future__ import annotations
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path('.').resolve()

TRAIN_JSON = PROJECT_ROOT / 'data/instances_attributes_train2020.json'
VAL_JSON = PROJECT_ROOT / 'data/instances_attributes_val2020.json'
TRAIN_IMAGES = PROJECT_ROOT / 'data/train'
VAL_IMAGES = PROJECT_ROOT / 'data/validation'

DATA_DIR = PROJECT_ROOT / 'data'
YOLO_DATASET_DIR = PROJECT_ROOT / 'yolo_dataset/'

OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'task3'

MODEL_NAME = 'yolov8n-seg.pt'
SEED = 42
IMG_SIZE = 384

assert YOLO_DATASET_DIR.exists()

run_name = 'yolo_seg_res'
model = YOLO(MODEL_NAME)

train_results = model.train(
    data=str(YOLO_DATASET_DIR / "fashionpedia_yolo_seg.yaml"),
    seed=SEED,
    imgsz=IMG_SIZE,
    project=str(OUTPUT_DIR / 'runs'),
    name=run_name,
    epochs=50,
    val=True,
    batch=-1,
    cache="disk",
    patience=10,
    save_period=1
)

print('Training finished.')
print(f'Train run dir: {train_results.save_dir}')
