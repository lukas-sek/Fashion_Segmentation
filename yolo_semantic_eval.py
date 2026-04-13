from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from ultralytics import YOLO

from experiment_utils import append_row_to_csv, ensure_dir, save_json, save_per_class_metrics
from segmentation_dataset import SegmentationRecord, _build_records_from_json, build_category_maps
from train_eval import compute_per_class_metrics


def _records_for_val_split(
    val_json: Path,
    val_image_dir: Path,
    val_mask_dir: Path,
) -> List[SegmentationRecord]:
    return _build_records_from_json(val_json, val_image_dir, val_mask_dir)


def _resize_rgb_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match Task 2 val: square resize (albumentations Resize on both image and mask)."""
    pil_im = Image.fromarray(image)
    pil_m = Image.fromarray(mask)
    pil_im = pil_im.resize((size, size), Image.BILINEAR)
    pil_m = pil_m.resize((size, size), Image.NEAREST)
    return np.array(pil_im), np.asarray(pil_m, dtype=np.int64)


def _instances_to_semantic_map(result, num_semantic_classes: int) -> torch.Tensor:
    """
    Rasterize YOLO instance masks to a dense semantic label map:
    semantic_id = yolo_class_id + 1 (0 = background), matching segmentation_dataset.build_category_maps.
    Overlaps: lower-confidence instances drawn first, higher confidence overwrites.
    """
    h, w = int(result.orig_shape[0]), int(result.orig_shape[1])
    pred = torch.zeros((h, w), dtype=torch.long)
    if result.boxes is None or result.masks is None or len(result.boxes) == 0:
        return pred

    conf = result.boxes.conf.detach().cpu()
    cls = result.boxes.cls.detach().cpu().long()
    md = result.masks.data.detach().cpu()  # n, h, w uint8/float

    order = torch.argsort(conf)  # ascending: paint high-conf last
    for idx in order:
        i = int(idx.item())
        binary = md[i] > 0
        c = int(cls[i].item())
        sem = c + 1
        if sem < 0 or sem >= num_semantic_classes:
            continue
        pred[binary] = sem
    return pred


@torch.no_grad()
def evaluate_yolo(
    weights: Path,
    val_json: Path,
    val_image_dir: Path,
    val_mask_dir: Path,
    imgsz: int,
    annotation_json_for_class_names: Path,
    device: Optional[str] = None,
    conf: float = 0.25,
    max_samples: Optional[int] = None,
) -> Dict:
    """
    Returns dict with val_pixel_acc_ex_bg, val_mdice_ex_bg, val_miou_ex_bg, per_class_metrics, confusion_matrix.
    """
    _, semantic_id_to_name = build_category_maps(
        annotation_json_for_class_names)
    num_classes = len(semantic_id_to_name)
    class_names = dict(sorted(semantic_id_to_name.items()))

    records = _records_for_val_split(val_json, val_image_dir, val_mask_dir)
    if max_samples is not None:
        records = records[: max(0, int(max_samples))]

    model = YOLO(str(weights))
    dev = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    pixel_accuracy = MulticlassAccuracy(
        num_classes=num_classes,
        average="micro",
        multidim_average="global",
        ignore_index=0,
    ).to(dev)
    confusion = MulticlassConfusionMatrix(num_classes=num_classes).to(dev)

    # Ultralytics expects str or None for device
    predict_device = device if device is not None else (
        0 if torch.cuda.is_available() else "cpu")

    t0 = time.perf_counter()
    processed = 0
    for rec in records:
        image = np.array(Image.open(rec.image_path).convert("RGB"))
        mask = np.array(Image.open(rec.mask_path), dtype=np.uint8)
        image, target = _resize_rgb_and_mask(image, mask, imgsz)

        results = model.predict(
            source=image,
            conf=conf,
            verbose=False,
        )
        pred_map = _instances_to_semantic_map(results[0], num_classes)

        pred_b = pred_map.unsqueeze(0).to(dev)
        tgt_b = torch.from_numpy(target).unsqueeze(0).to(dev)

        pixel_accuracy.update(pred_b, tgt_b)
        confusion.update(pred_b, tgt_b)
        processed += 1

    elapsed = time.perf_counter() - t0

    confmat = confusion.compute().detach().cpu()
    per_class_rows, mdice_ex_bg, miou_ex_bg = compute_per_class_metrics(
        confmat=confmat,
        class_names=class_names,
        background_index=0,
    )

    return {
        "num_val_images": processed,
        "seconds": float(elapsed),
        "val_pixel_acc_ex_bg": float(pixel_accuracy.compute().detach().cpu().item()),
        "val_mdice_ex_bg": mdice_ex_bg,
        "val_miou_ex_bg": miou_ex_bg,
        "per_class_metrics": per_class_rows,
        "confusion_matrix": confmat.numpy(),
        "num_classes": num_classes,
        "imgsz": imgsz,
        "conf_threshold": conf,
    }


def save_yolo_outputs(
    metrics: Dict,
    *,
    run_name: str,
    weights_path: Path,
    output_dir: Path,
    extra_summary_fields: Optional[Dict] = None,
) -> Dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    metrics_dir = ensure_dir(output_dir / "per_class_metrics")
    configs_dir = ensure_dir(output_dir / "configs")

    per_class_path = metrics_dir / f"{run_name}.csv"
    save_per_class_metrics(metrics["per_class_metrics"], per_class_path)

    config_path = configs_dir / f"{run_name}.json"
    config_payload = {
        "run_name": run_name,
        "task": "yolo_instance_seg_semantic_eval",
        "weights": str(weights_path),
        "imgsz": metrics["imgsz"],
        "conf_threshold": metrics["conf_threshold"],
        "num_classes": metrics["num_classes"],
        "num_val_images": metrics["num_val_images"],
        "eval_seconds": metrics["seconds"],
        "note": "Dense pixel metrics align with train_eval.validate_one_epoch (Task 2). "
        "YOLO seg mAP from model.val() is reported separately and is not the same quantity.",
    }
    if extra_summary_fields:
        config_payload.update(extra_summary_fields)
    save_json(config_path, config_payload)

    json_metrics_path = output_dir / f"{run_name}_metrics.json"
    save_json(
        json_metrics_path,
        {
            "val_pixel_acc_ex_bg": metrics["val_pixel_acc_ex_bg"],
            "val_mdice_ex_bg": metrics["val_mdice_ex_bg"],
            "val_miou_ex_bg": metrics["val_miou_ex_bg"],
            "seg_map50": (extra_summary_fields or {}).get("seg_map50"),
            "seg_map50_95": (extra_summary_fields or {}).get("seg_map50_95"),
            "box_map50": (extra_summary_fields or {}).get("box_map50"),
            "box_map50_95": (extra_summary_fields or {}).get("box_map50_95"),
            "num_val_images": metrics["num_val_images"],
            "imgsz": metrics["imgsz"],
            "conf_threshold": metrics["conf_threshold"],
        },
    )

    return {
        "per_class_csv": per_class_path,
        "config_json": config_path,
        "metrics_json": json_metrics_path,
    }
