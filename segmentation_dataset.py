from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SegmentationRecord:
    image_id: int
    image_name: str
    image_path: str
    mask_path: str


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_category_maps(annotation_json: str | Path) -> tuple[Dict[int, str], Dict[int, str]]:
    data = load_json(annotation_json)
    category_id_to_name = {
        int(category["id"]): str(category["name"])
        for category in data.get("categories", [])
    }
    semantic_id_to_name = {0: "background"}
    semantic_id_to_name.update(
        {category_id + 1: name for category_id, name in category_id_to_name.items()}
    )
    return category_id_to_name, semantic_id_to_name


def _build_records_from_json(
    annotation_json: str | Path,
    image_dir: str | Path,
    mask_dir: str | Path,
) -> List[SegmentationRecord]:
    data = load_json(annotation_json)
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    records: List[SegmentationRecord] = []

    for image_info in data.get("images", []):
        image_name = image_info["file_name"]
        stem = Path(image_name).stem
        image_path = image_dir / image_name
        mask_path = mask_dir / f"{stem}_seg.png"
        if image_path.exists() and mask_path.exists():
            records.append(
                SegmentationRecord(
                    image_id=int(image_info["id"]),
                    image_name=image_name,
                    image_path=str(image_path),
                    mask_path=str(mask_path),
                )
            )
    return records


def get_train_val_records(
    train_json_path: str | Path,
    val_json_path: str | Path,
    train_image_dir: str | Path = "data/train",
    val_image_dir: str | Path = "data/test",
    train_mask_dir: str | Path = "data/segmentations_train",
    val_mask_dir: str | Path = "data/segmentations_val",
) -> tuple[List[SegmentationRecord], List[SegmentationRecord]]:
    train_records = _build_records_from_json(train_json_path, train_image_dir, train_mask_dir)
    val_records = _build_records_from_json(val_json_path, val_image_dir, val_mask_dir)
    return train_records, val_records


def subset_records(records: Iterable[SegmentationRecord], max_items: Optional[int] = None) -> List[SegmentationRecord]:
    records = list(records)
    if max_items is None:
        return records
    return records[: max(0, int(max_items))]


def remap_mask(mask: np.ndarray, label_remap: Optional[Dict[int, int]] = None) -> np.ndarray:
    if label_remap is None:
        return mask.astype(np.int64, copy=False)

    remapped = np.zeros_like(mask, dtype=np.int64)
    for source_label, target_label in label_remap.items():
        remapped[mask == int(source_label)] = int(target_label)
    return remapped


def make_reduced_label_remap(keep_semantic_ids: Iterable[int], background_index: int = 0) -> Dict[int, int]:
    keep_semantic_ids = sorted({int(label) for label in keep_semantic_ids if int(label) != background_index})
    label_remap = {background_index: background_index}
    for new_label, semantic_id in enumerate(keep_semantic_ids, start=1):
        label_remap[semantic_id] = new_label
    return label_remap


def apply_label_remap_to_names(
    semantic_id_to_name: Dict[int, str],
    label_remap: Optional[Dict[int, int]],
    background_index: int = 0,
) -> Dict[int, str]:
    if label_remap is None:
        return dict(sorted(semantic_id_to_name.items()))

    new_name_map = {background_index: semantic_id_to_name.get(background_index, "background")}
    for old_label, new_label in sorted(label_remap.items(), key=lambda item: item[1]):
        if old_label == background_index or new_label == background_index:
            continue
        new_name_map[new_label] = semantic_id_to_name.get(old_label, f"class_{old_label}")
    return dict(sorted(new_name_map.items()))


class FashionSegmentationDataset(Dataset):
    def __init__(
        self,
        records: List[SegmentationRecord],
        image_dir: str | Path | None = None,
        mask_dir: str | Path | None = None,
        transform: Optional[Callable] = None,
        label_remap: Optional[Dict[int, int]] = None,
    ) -> None:
        self.records = records
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.label_remap = label_remap

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image = np.array(Image.open(record.image_path).convert("RGB"))
        mask = np.array(Image.open(record.mask_path), dtype=np.uint8)
        mask = remap_mask(mask, self.label_remap)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.asarray(mask))

        return {
            "image": image.float(),
            "mask": mask.long(),
            "image_id": record.image_id,
            "image_name": record.image_name,
            "image_path": record.image_path,
            "mask_path": record.mask_path,
        }
