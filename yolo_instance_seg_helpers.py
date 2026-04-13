import json
import shutil
from pathlib import Path
from typing import Dict, Tuple
from ultralytics.data.converter import convert_coco
import yaml


def load_json(path: Path) -> Dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def name_list_from_coco_categories(coco_json: Dict) -> [str]:
    categories = sorted(coco_json.get('categories', []),
                        key=lambda x: int(x['id']))
    return [str(cat['name']) for cat in categories]


def copy_images(source_dir: Path, dest_dir: Path) -> Tuple[int, int]:
    ensure_dir(dest_dir)

    for src in sorted(source_dir.iterdir()):
        if not src.is_file():
            continue
        try:
            shutil.copy2(src, dest_dir / src.name)
        except Exception:
            raise RuntimeError(f'Failed to copy {src} to {dest_dir}')


def save_dataset_yaml(payload: Dict, path: Path) -> None:
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _is_valid_polygon(poly) -> bool:
    return isinstance(poly, list)


def sanitize_annotations_for_ultralytics(src_json: Path, dst_json: Path) -> Dict[str, int]:
    data = load_json(src_json)
    anns = data.get('annotations', [])
    categories = data.get('categories', [])

    kept = []
    dropped_non_list = 0
    dropped_invalid_poly = 0

    shifted_category_ids = 0

    for ann in anns:
        seg = ann.get('segmentation')
        if not isinstance(seg, list):
            dropped_non_list += 1
            continue

        valid_polys = [poly for poly in seg if _is_valid_polygon(poly)]
        if not valid_polys:
            dropped_invalid_poly += 1
            continue

        ann_new = dict(ann)
        # Fashionpedia category IDs are already 0-based. Ultralytics COCO conversion
        # expects 1-based COCO-style IDs and subtracts 1 internally. Shift to 1-based
        # here so resulting YOLO labels stay in [0, nc-1] instead of producing -1.
        ann_new['category_id'] = int(ann_new['category_id']) + 1
        shifted_category_ids += 1
        ann_new['segmentation'] = valid_polys
        kept.append(ann_new)

    shifted_categories = 0
    updated_categories = []
    for cat in categories:
        cat_new = dict(cat)
        cat_new['id'] = int(cat_new['id']) + 1
        shifted_categories += 1
        updated_categories.append(cat_new)

    data['annotations'] = kept
    if categories:
        data['categories'] = updated_categories
    with dst_json.open('w', encoding='utf-8') as f:
        json.dump(data, f)

    return {
        'annotations_total': len(anns),
        'annotations_kept': len(kept),
        'annotations_dropped_non_polygon_type': dropped_non_list,
        'annotations_dropped_invalid_polygon': dropped_invalid_poly,
        'category_ids_shifted_to_one_based': shifted_category_ids,
        'categories_shifted_to_one_based': shifted_categories,
    }


def create_yolo_formatted_dataset(DATA_DIR: Path, TRAIN_JSON: Path, VAL_JSON: Path, YOLO_DATASET_DIR: Path, TRAIN_IMAGES: Path, VAL_IMAGES: Path) -> None:
    if YOLO_DATASET_DIR.exists():
        print(
            f"Dataset directory already exists, skipping conversion: {YOLO_DATASET_DIR}")
        return

    ann_dir = ensure_dir(DATA_DIR / '_annotations_for_convert')
    train_sanitized = ann_dir / 'instances_train.json'
    val_sanitized = ann_dir / 'instances_val.json'

    sanitize_annotations_for_ultralytics(TRAIN_JSON, train_sanitized)
    sanitize_annotations_for_ultralytics(VAL_JSON, val_sanitized)

    convert_coco(
        labels_dir=ann_dir,
        save_dir=YOLO_DATASET_DIR,
        use_segments=True,
        use_keypoints=False,
        cls91to80=False,
        lvis=False,
    )

    copy_images(TRAIN_IMAGES, YOLO_DATASET_DIR / 'images' / 'train')
    copy_images(VAL_IMAGES, YOLO_DATASET_DIR / 'images' / 'val')

    train_coco = load_json(TRAIN_JSON)
    name_list = name_list_from_coco_categories(train_coco)
    payload = {
        'path': "./yolo_dataset/",
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(name_list),
        'names': name_list,
    }
    save_dataset_yaml(payload, YOLO_DATASET_DIR / "fashionpedia_yolo_seg.yaml")
