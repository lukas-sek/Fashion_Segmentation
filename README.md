# Fashion Segmentation — Fashionpedia

Semantic and instance segmentation on the [Fashionpedia](https://fashionpedia.github.io/home/) dataset (46 apparel categories + background). This project implements three deliverable tasks: exploratory data analysis, dense semantic segmentation with multiple architectures, and YOLO-based instance segmentation.

## Tasks

### Task 1 — Exploratory Analysis

Per-category pixel statistics (coverage ratio, frequency, variance) computed from rasterised ground-truth masks. Results are saved to `label_distribution_stats.csv` and visualised in the early sections of `fashion-parsing.ipynb`.

### Task 2 — Semantic Segmentation

Trains and evaluates three encoder–decoder architectures via [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch):

| Model | Encoder | Pretrained |
|-------|---------|------------|
| DeepLabV3+ | ResNet-50 | ImageNet |
| SegFormer | MiT-B0 | ImageNet |
| UPerNet | Swin-Tiny | ImageNet |

Experiments sweep over resolution (192 / 384), learning rate, batch size, and augmentation strength (weak / strong). A reduced-label finetuning study drops over-represented classes and continues from a full-label checkpoint.

**Metrics:** micro pixel accuracy (ex-bg), mean Dice (ex-bg), mean IoU (ex-bg), per-class Dice and IoU — all computed from the confusion matrix with background excluded.

### Task 3 — Instance Segmentation (YOLO)

Converts Fashionpedia COCO-style polygons into YOLO segmentation labels and trains **YOLOv8n-seg** with Ultralytics. Evaluation reports both native instance metrics (box/mask mAP@0.5, mAP@0.5:0.95) and Task 2–aligned dense pixel metrics for cross-architecture comparison.

## Repository Structure

```
Fashion_Segmentation/
├── fashion-parsing.ipynb               # Task 1 analysis + Task 2 semantic segmentation
├── yolo_instance_segmentation.ipynb    # Task 3 YOLO instance segmentation
│
├── model_factory.py                    # Builds DeepLabV3+, SegFormer, UPerNet via SMP
├── segmentation_dataset.py             # Dataset, category maps, label remapping
├── augmentations.py                    # Albumentations presets (weak / strong / val)
├── train_eval.py                       # Training loop, validation, per-class metrics
├── experiment_utils.py                 # Seeding, run naming, CSV/JSON helpers
│
├── yolo_instance_seg_helpers.py        # COCO → YOLO format conversion
├── yolo_semantic_eval.py               # Dense pixel evaluation for YOLO predictions
├── create_validation_only_folder.py    # Copies val images into data/validation/
│
├── requirements.txt                    # Python dependencies
```

## Setup

### Prerequisites

- Python 3.10+
- CUDA 12.1 (for GPU training) or CPU-only PyTorch

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data Preparation

1. Download the Fashionpedia dataset (train/val images and annotation JSONs) and place them under `data/`:

```
data/
├── instances_attributes_train2020.json
├── instances_attributes_val2020.json
├── train/          # training images
└── test/           # test/validation images
```

2. Rasterise polygon annotations into semantic mask PNGs using the bundled COCO API tool:

```bash
cd cocoapi/PythonAPI
python fashionpedia.py ../../data/instances_attributes_train2020.json ../../data/segmentations_train
python fashionpedia.py ../../data/instances_attributes_val2020.json ../../data/segmentations_val
```

## Usage

### Task 1 + Task 2

Open and run `fashion-parsing.ipynb`. The notebook is structured sequentially:

- **Sections 1–9:** exploratory analysis — category frequencies, pixel coverage, dominance plots.
- **Task 2 cells:** configure run mode (`smoke` / `pilot` / `full`), execute the experiment grid, and review `outputs/task2/run_summary.csv`.
- **Reduced-label study:** select overrepresented categories, remap labels, finetune from a full-label checkpoint.

### Task 3

Open and run `yolo_instance_segmentation.ipynb`:

1. The notebook converts the dataset to YOLO format (or reuses an existing `yolo_dataset/` folder).
2. Trains YOLOv8n-seg and saves run artifacts under `outputs/task3/runs/`.
3. Evaluates with both Ultralytics mAP and dense pixel metrics aligned with Task 2.

## Metrics Reference

| Metric | Scope | Source |
|--------|-------|--------|
| Pixel Accuracy (ex-bg) | Micro, background excluded | `torchmetrics.MulticlassAccuracy` |
| Mean Dice (ex-bg) | Macro over foreground classes | Confusion matrix TP/FP/FN |
| Mean IoU (ex-bg) | Macro over foreground classes | Confusion matrix TP/FP/FN |
| Box mAP@0.5 / mAP@0.5:0.95 | Instance detection | Ultralytics `model.val()` |
| Mask mAP@0.5 / mAP@0.5:0.95 | Instance segmentation | Ultralytics `model.val()` |

Dense pixel metrics (Dice, IoU, pixel accuracy) are directly comparable across Task 2 and Task 3. YOLO mAP is an instance-level metric and is reported separately.
