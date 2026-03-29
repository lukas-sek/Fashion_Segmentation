from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image


MODEL_SPECS = {
    "deeplabv3plus_r50": {
        "family": "cnn",
        "display_name": "DeepLabV3+ (ResNet-50)",
        "justification": "Strong CNN baseline, stable, widely used, and a good reference point.",
        "base_config_candidates": [
            "configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb2-80k_ade20k-512x512.py",
            "configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-80k_ade20k-512x512.py",
            "configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_80k_ade20k.py",
        ],
    },
    "segformer_mit-b0": {
        "family": "transformer",
        "display_name": "SegFormer (MiT-B0)",
        "justification": "Modern transformer-style lightweight model with a strong efficiency/accuracy trade-off.",
        "base_config_candidates": [
            "configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py",
            "configs/segformer/segformer_mit-b0_512x512_160k_ade20k.py",
        ],
    },
    "upernet_swin-tiny": {
        "family": "transformer",
        "display_name": "UPerNet (Swin-Tiny)",
        "justification": "Stronger hierarchical transformer model that complements SegFormer and usually offers higher capacity.",
        "base_config_candidates": [
            "configs/swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py",
            "configs/swin/swin-tiny-patch4-window7_upernet_8xb2-160k_ade20k-512x512.py",
            "configs/upernet/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py",
        ],
    },
}

LR_CANDIDATES = [3e-5, 1e-4, 3e-4]
BATCH_SIZE_CANDIDATES = [4, 8, 16]
RESOLUTION_CANDIDATES = [(192, 192), (384, 384)]
AUGMENTATION_BRANCHES = ["basic"]


def build_palette(num_classes: int) -> list[list[int]]:
    palette = []
    for index in range(num_classes):
        palette.append([
            (37 * index) % 256,
            (67 * index) % 256,
            (97 * index) % 256,
        ])
    return palette


def load_fashionpedia_classes(annotation_json_path: str | os.PathLike[str]) -> tuple[list[str], list[list[int]], pd.DataFrame]:
    with open(annotation_json_path, "r", encoding="utf-8") as handle:
        annotation_data = json.load(handle)

    categories = pd.DataFrame(annotation_data["categories"]).sort_values("id").reset_index(drop=True)
    classes = ["background"] + categories["name"].tolist()
    palette = build_palette(len(classes))
    return classes, palette, categories


def _find_image_path(image_dir: Path, file_name: str, valid_suffixes: Iterable[str]) -> Path | None:
    direct = image_dir / file_name
    if direct.exists():
        return direct

    stem = Path(file_name).stem
    for suffix in valid_suffixes:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def build_split_manifest(
    annotation_json_path: str | os.PathLike[str],
    image_dir: str | os.PathLike[str],
    mask_dir: str | os.PathLike[str],
    split_name: str,
    valid_suffixes: Iterable[str] = (".jpg", ".jpeg", ".png"),
) -> pd.DataFrame:
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    with open(annotation_json_path, "r", encoding="utf-8") as handle:
        annotation_data = json.load(handle)

    rows = []
    for image_info in annotation_data["images"]:
        image_path = _find_image_path(image_dir, image_info["file_name"], valid_suffixes)
        mask_path = mask_dir / f"{Path(image_info['file_name']).stem}_seg.png"
        if image_path is None or not mask_path.exists():
            continue

        rows.append({
            "split": split_name,
            "image_id": image_info["id"],
            "stem": image_path.stem,
            "image_name": image_path.name,
            "image_path": str(image_path.resolve()),
            "mask_path": str(mask_path.resolve()),
            "height": image_info.get("height"),
            "width": image_info.get("width"),
            "image_suffix": image_path.suffix.lower(),
        })

    manifest_df = pd.DataFrame(rows).sort_values("stem").reset_index(drop=True)
    if manifest_df.empty:
        raise ValueError(f"No valid image/mask pairs found for split '{split_name}'.")
    return manifest_df


def write_manifest_artifacts(
    train_manifest: pd.DataFrame,
    val_manifest: pd.DataFrame,
    output_dir: str | os.PathLike[str],
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_manifest = pd.concat([train_manifest, val_manifest], ignore_index=True)
    paths = {
        "train_manifest": output_dir / "train_manifest.csv",
        "val_manifest": output_dir / "val_manifest.csv",
        "all_manifest": output_dir / "all_manifest.csv",
        "train_split": output_dir / "train.txt",
        "val_split": output_dir / "val.txt",
    }

    train_manifest.to_csv(paths["train_manifest"], index=False)
    val_manifest.to_csv(paths["val_manifest"], index=False)
    all_manifest.to_csv(paths["all_manifest"], index=False)
    paths["train_split"].write_text("\n".join(train_manifest["stem"]) + "\n", encoding="utf-8")
    paths["val_split"].write_text("\n".join(val_manifest["stem"]) + "\n", encoding="utf-8")
    return paths


def sample_manifest(
    manifest_df: pd.DataFrame,
    *,
    fraction: float | None = None,
    max_items: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    if manifest_df.empty:
        return manifest_df.copy()

    sampled_df = manifest_df.copy()

    if fraction is not None:
        if not (0 < fraction <= 1):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        n_items = max(1, int(round(len(sampled_df) * fraction)))
        sampled_df = sampled_df.sample(n=n_items, random_state=random_seed)

    if max_items is not None:
        if max_items <= 0:
            raise ValueError(f"max_items must be positive, got {max_items}")
        if len(sampled_df) > max_items:
            sampled_df = sampled_df.sample(n=max_items, random_state=random_seed)

    return sampled_df.sort_values("stem").reset_index(drop=True)


def summarize_model_choices() -> pd.DataFrame:
    rows = []
    for model_key, spec in MODEL_SPECS.items():
        rows.append({
            "model": model_key,
            "display_name": spec["display_name"],
            "family": spec["family"],
            "justification": spec["justification"],
        })
    return pd.DataFrame(rows)


def pick_dominant_labels(
    label_stats_csv: str | os.PathLike[str],
    top_k: int = 5,
) -> pd.DataFrame:
    stats_df = pd.read_csv(label_stats_csv)
    dominant_df = stats_df.sort_values("mean_ratio_full", ascending=False).head(top_k).copy()
    dominant_df["mask_value"] = dominant_df["category_id"] + 1
    return dominant_df.reset_index(drop=True)


def build_reduced_label_mapping(
    categories_df: pd.DataFrame,
    ignored_category_ids: Iterable[int],
) -> pd.DataFrame:
    ignored_ids = set(int(x) for x in ignored_category_ids)

    rows = [{
        "original_category_id": -1,
        "original_mask_value": 0,
        "category_name": "background",
        "status": "kept",
        "new_mask_value": 0,
    }]

    next_label = 1
    for row in categories_df.sort_values("id").itertuples(index=False):
        category_id = int(row.id)
        original_mask_value = category_id + 1
        if category_id in ignored_ids:
            rows.append({
                "original_category_id": category_id,
                "original_mask_value": original_mask_value,
                "category_name": row.name,
                "status": "ignored",
                "new_mask_value": 255,
            })
        else:
            rows.append({
                "original_category_id": category_id,
                "original_mask_value": original_mask_value,
                "category_name": row.name,
                "status": "kept",
                "new_mask_value": next_label,
            })
            next_label += 1

    return pd.DataFrame(rows)


def remap_mask_array(mask_array: np.ndarray, mapping_df: pd.DataFrame) -> np.ndarray:
    mapping = {
        int(row.original_mask_value): int(row.new_mask_value)
        for row in mapping_df.itertuples(index=False)
    }
    remapped = np.full(mask_array.shape, 255, dtype=np.uint8)
    for original_value, new_value in mapping.items():
        remapped[mask_array == original_value] = new_value
    return remapped


def export_reduced_masks(
    manifest_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    output_mask_dir: str | os.PathLike[str],
) -> pd.DataFrame:
    output_mask_dir = Path(output_mask_dir)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    remapped_rows = []
    for row in manifest_df.itertuples(index=False):
        mask = np.array(Image.open(row.mask_path), dtype=np.uint8)
        remapped = remap_mask_array(mask, mapping_df)
        output_path = output_mask_dir / Path(row.mask_path).name
        Image.fromarray(remapped, mode="L").save(output_path)

        updated = dict(row._asdict())
        updated["mask_path"] = str(output_path.resolve())
        remapped_rows.append(updated)

    return pd.DataFrame(remapped_rows)


def resolve_base_config(mmseg_root: str | os.PathLike[str], model_key: str) -> str:
    mmseg_root = Path(mmseg_root)
    spec = MODEL_SPECS[model_key]
    for relative_path in spec["base_config_candidates"]:
        candidate = mmseg_root / relative_path
        if candidate.exists():
            return str(candidate.resolve())

    candidate_list = "\n".join(f"- {path}" for path in spec["base_config_candidates"])
    raise FileNotFoundError(
        f"Could not resolve a base config for '{model_key}' under '{mmseg_root}'. Tried:\n{candidate_list}"
    )


def _format_python_list(items: Iterable[str]) -> str:
    return "[" + ", ".join(repr(item) for item in items) + "]"


def build_train_pipeline_config(
    resolution: tuple[int, int],
    augmentation_branch: str,
    manifest_path: str,
) -> str:
    height, width = resolution
    pipeline = [
        "train_pipeline = [",
        "    dict(type='LoadImageFromFile'),",
        "    dict(type='LoadAnnotations', reduce_zero_label=False),",
        f"    dict(type='RandomResize', scale=({width}, {height}), ratio_range=(0.8, 1.2), keep_ratio=False),",
        "    dict(type='RandomFlip', prob=0.5),",
        "    dict(type='RandomRotate', prob=0.3, degree=10, pad_val=0, seg_pad_val=255),",
        f"    dict(type='RandomCrop', crop_size=({height}, {width}), cat_max_ratio=0.9),",
        "    dict(type='PhotoMetricDistortion'),",
        "    dict(type='FashionSegCutOut', prob=0.3, num_holes=2, min_ratio=0.05, max_ratio=0.12, fill_value=0),",
        "    dict(type='PackSegInputs'),",
        "]",
    ]
    return "\n".join(pipeline)


def build_test_pipeline_config(resolution: tuple[int, int]) -> str:
    height, width = resolution
    return "\n".join([
        "test_pipeline = [",
        "    dict(type='LoadImageFromFile'),",
        "    dict(type='LoadAnnotations', reduce_zero_label=False),",
        f"    dict(type='Resize', scale=({width}, {height}), keep_ratio=False),",
        "    dict(type='PackSegInputs'),",
        "]",
    ])


def _make_dataset_block(
    image_dir: str,
    mask_dir: str,
    split_file: str,
    img_suffix: str,
    classes: list[str],
    palette: list[list[int]],
    pipeline_name: str,
) -> str:
    return f"""dict(
        type='BaseSegDataset',
        data_root='',
        ann_file=r'{split_file}',
        data_prefix=dict(img_path=r'{image_dir}', seg_map_path=r'{mask_dir}'),
        img_suffix='{img_suffix}',
        seg_map_suffix='_seg.png',
        metainfo=dict(classes={classes!r}, palette={palette!r}),
        reduce_zero_label=False,
        pipeline={pipeline_name},
    )"""


def generate_config_text(
    *,
    mmseg_root: str | os.PathLike[str],
    project_root: str | os.PathLike[str],
    model_key: str,
    experiment_name: str,
    train_image_dir: str,
    train_mask_dir: str,
    val_image_dir: str,
    val_mask_dir: str,
    train_split_file: str,
    val_split_file: str,
    train_manifest_path: str,
    img_suffix: str,
    classes: list[str],
    palette: list[list[int]],
    resolution: tuple[int, int],
    lr: float,
    batch_size: int,
    augmentation_branch: str,
    max_iters: int,
    val_interval: int,
    work_dir: str,
) -> str:
    base_config = resolve_base_config(mmseg_root, model_key)
    num_classes = len(classes)
    train_pipeline = build_train_pipeline_config(resolution, augmentation_branch, train_manifest_path)
    test_pipeline = build_test_pipeline_config(resolution)

    return f"""_base_ = [r'{base_config}']

custom_imports = dict(
    imports=['fashion_mmseg_transforms'],
    allow_failed_imports=False,
)

crop_size = {resolution!r}
num_classes = {num_classes}
work_dir = r'{work_dir}'

{train_pipeline}

{test_pipeline}

train_dataloader = dict(
    batch_size={batch_size},
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset={_make_dataset_block(train_image_dir, train_mask_dir, train_split_file, img_suffix, classes, palette, 'train_pipeline')}
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset={_make_dataset_block(val_image_dir, val_mask_dir, val_split_file, img_suffix, classes, palette, 'test_pipeline')}
)

test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice'])
test_evaluator = val_evaluator

model = dict(
    data_preprocessor=dict(size=crop_size),
    decode_head=dict(num_classes=num_classes, ignore_index=255),
)

if 'auxiliary_head' in model:
    if isinstance(model['auxiliary_head'], list):
        for head in model['auxiliary_head']:
            head['num_classes'] = num_classes
            head['ignore_index'] = 255
    else:
        model['auxiliary_head']['num_classes'] = num_classes
        model['auxiliary_head']['ignore_index'] = 255

optim_wrapper = dict(optimizer=dict(lr={lr}))
train_cfg = dict(type='IterBasedTrainLoop', max_iters={max_iters}, val_interval={val_interval})
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval={val_interval}, save_best='mDice'),
    logger=dict(interval=50),
)
randomness = dict(seed=42)
"""


def write_config_file(config_text: str, config_path: str | os.PathLike[str]) -> Path:
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_text, encoding="utf-8")
    return config_path


def build_stage1_lr_sweep(schedule_name: str = "ranking_short") -> pd.DataFrame:
    rows = []
    for model_key in MODEL_SPECS:
        for resolution in RESOLUTION_CANDIDATES:
            for lr in LR_CANDIDATES:
                rows.append({
                    "phase": "stage1_lr",
                    "label_set": "full",
                    "model": model_key,
                    "resolution": resolution,
                    "lr": lr,
                    "batch_size": None,
                    "augmentation_branch": "basic",
                    "schedule": schedule_name,
                    "checkpoint": None,
                    "mDice": None,
                    "accuracy": None,
                })
    return pd.DataFrame(rows)


def build_stage2_batch_sweep(best_lr_by_model_res: dict[tuple[str, tuple[int, int]], float], schedule_name: str = "ranking_short") -> pd.DataFrame:
    rows = []
    for (model_key, resolution), lr in best_lr_by_model_res.items():
        for batch_size in BATCH_SIZE_CANDIDATES:
            rows.append({
                "phase": "stage2_batch",
                "label_set": "full",
                "model": model_key,
                "resolution": resolution,
                "lr": lr,
                "batch_size": batch_size,
                "augmentation_branch": "basic",
                "schedule": schedule_name,
                "checkpoint": None,
                "mDice": None,
                "accuracy": None,
            })
    return pd.DataFrame(rows)


def build_stage4_final_confirmation(
    final_params_by_model_res: dict[tuple[str, tuple[int, int]], dict[str, float | int | str]],
    schedule_name: str = "final_long",
) -> pd.DataFrame:
    rows = []
    for (model_key, resolution), params in final_params_by_model_res.items():
        rows.append({
            "phase": "stage4_final",
            "label_set": "full",
            "model": model_key,
            "resolution": resolution,
            "lr": params["lr"],
            "batch_size": params["batch_size"],
            "augmentation_branch": params.get("augmentation_branch", "basic"),
            "schedule": schedule_name,
            "checkpoint": None,
            "mDice": None,
            "accuracy": None,
        })
    return pd.DataFrame(rows)


def build_reduced_label_experiments(
    *,
    best_model: str,
    best_resolution: tuple[int, int],
    best_lr: float,
    best_batch_size: int,
    schedule_name: str = "ranking_short",
) -> pd.DataFrame:
    rows = []
    for lr in LR_CANDIDATES:
        rows.append({
            "phase": "reduced_stage1_lr",
            "label_set": "reduced_top5_ignore",
            "model": best_model,
            "resolution": best_resolution,
            "lr": lr,
            "batch_size": best_batch_size,
            "augmentation_branch": "basic",
            "schedule": schedule_name,
            "checkpoint": None,
            "mDice": None,
            "accuracy": None,
        })

    for batch_size in BATCH_SIZE_CANDIDATES:
        rows.append({
            "phase": "reduced_stage2_batch",
            "label_set": "reduced_top5_ignore",
            "model": best_model,
            "resolution": best_resolution,
            "lr": best_lr,
            "batch_size": batch_size,
            "augmentation_branch": "basic",
            "schedule": schedule_name,
            "checkpoint": None,
            "mDice": None,
            "accuracy": None,
        })

    return pd.DataFrame(rows)


def write_experiment_manifest(experiments_df: pd.DataFrame, output_path: str | os.PathLike[str]) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    experiments_df.to_csv(output_path, index=False)
    return output_path


def build_train_command(
    *,
    mmseg_root: str | os.PathLike[str],
    project_root: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
) -> str:
    mmseg_root = Path(mmseg_root).resolve()
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    return (
        f'$env:PYTHONPATH="{project_root};{mmseg_root}"; '
        f'& python "{(mmseg_root / "tools" / "train.py").resolve()}" "{config_path}"'
    )


def collect_scalar_metrics(work_dir: str | os.PathLike[str], experiment_name: str) -> dict[str, float | int | None]:
    scalars_path = Path(work_dir) / experiment_name / "vis_data" / "scalars.json"
    if not scalars_path.exists():
        return {"mDice": None, "accuracy": None, "mAcc": None, "aAcc": None, "step": None}

    rows = []
    with open(scalars_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    metric_rows = [row for row in rows if "mDice" in row]
    if not metric_rows:
        return {"mDice": None, "accuracy": None, "mAcc": None, "aAcc": None, "step": None}

    best_row = max(metric_rows, key=lambda row: row.get("mDice", float("-inf")))
    return {
        "mDice": best_row.get("mDice"),
        "accuracy": best_row.get("aAcc"),
        "mAcc": best_row.get("mAcc"),
        "aAcc": best_row.get("aAcc"),
        "step": best_row.get("step"),
    }


def build_resolution_delta_table(
    per_class_results_df: pd.DataFrame,
    label_stats_df: pd.DataFrame,
    metric_col: str = "mDice",
) -> pd.DataFrame:
    subset = per_class_results_df[per_class_results_df["resolution"].isin(["192x192", "384x384"])].copy()
    pivot_df = subset.pivot_table(
        index=["model", "label"],
        columns="resolution",
        values=metric_col,
        aggfunc="first",
    ).reset_index()
    pivot_df["delta_metric"] = pivot_df["384x384"] - pivot_df["192x192"]
    merged = pivot_df.merge(
        label_stats_df.rename(columns={"category_name": "label"}),
        on="label",
        how="left",
    )
    return merged.sort_values(["model", "delta_metric"], ascending=[True, False]).reset_index(drop=True)
