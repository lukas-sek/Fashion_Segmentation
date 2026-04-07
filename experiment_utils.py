from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def flatten_config_for_name(config: Dict) -> str:
    return (
        f"{config['model_name']}"
        f"_res{config['image_size']}"
        f"_lr{config['learning_rate']}"
        f"_bs{config['batch_size']}"
        f"_{config['augmentation']}"
        f"_{config['run_mode']}"
        f"_{config.get('reduced_label_tag', 'full_labels')}"
    ).replace("/", "-")


def append_row_to_csv(csv_path: str | Path, row: Dict) -> None:
    csv_path = Path(csv_path)
    frame = pd.DataFrame([row])
    if csv_path.exists():
        frame.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        frame.to_csv(csv_path, index=False)


def save_json(path: str | Path, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_history(history: List[Dict], path: str | Path) -> None:
    pd.DataFrame(history).to_csv(path, index=False)


def save_per_class_metrics(per_class_rows: Iterable[Dict], path: str | Path) -> None:
    pd.DataFrame(list(per_class_rows)).to_csv(path, index=False)


def load_results_table(results_dir: str | Path) -> pd.DataFrame:
    summary_path = Path(results_dir) / "run_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()
    return pd.read_csv(summary_path)


def build_run_paths(output_root: str | Path, config: Dict) -> Dict[str, Path]:
    output_root = ensure_dir(output_root)
    run_name = flatten_config_for_name(config)
    checkpoints_dir = ensure_dir(output_root / "checkpoints")
    histories_dir = ensure_dir(output_root / "histories")
    metrics_dir = ensure_dir(output_root / "per_class_metrics")
    configs_dir = ensure_dir(output_root / "configs")

    return {
        "run_name": run_name,
        "summary_csv": output_root / "run_summary.csv",
        "checkpoint": checkpoints_dir / f"{run_name}.pt",
        "history_csv": histories_dir / f"{run_name}.csv",
        "per_class_csv": metrics_dir / f"{run_name}.csv",
        "config_json": configs_dir / f"{run_name}.json",
    }


def timer() -> float:
    return time.perf_counter()


def elapsed_seconds(start_time: float) -> float:
    return time.perf_counter() - start_time


def select_overrepresented_labels(
    stats_df: pd.DataFrame,
    rule: str = "ratio_freq_combo",
    category_id_column: str = "category_id",
    frequency_column: str = "num_images",
    ratio_column: str = "mean_ratio_full",
    threshold_quantile: float = 0.85,
) -> pd.DataFrame:
    if stats_df.empty:
        raise ValueError("stats_df is empty.")
    if rule != "ratio_freq_combo":
        raise ValueError(f"Unsupported rule: {rule}")

    df = stats_df.dropna(subset=[frequency_column, ratio_column]).copy()
    freq = df[frequency_column].astype(float)
    ratio = df[ratio_column].astype(float)
    df["norm_num_images"] = (freq - freq.min()) / (freq.max() - freq.min() + 1e-8)
    df["norm_mean_ratio_full"] = (ratio - ratio.min()) / (ratio.max() - ratio.min() + 1e-8)
    df["overrep_score"] = 0.5 * df["norm_num_images"] + 0.5 * df["norm_mean_ratio_full"]
    cutoff = df["overrep_score"].quantile(threshold_quantile)
    df["drop_for_reduced_label_run"] = df["overrep_score"] >= cutoff
    df["semantic_id"] = df[category_id_column].astype(int) + 1
    return df.sort_values("overrep_score", ascending=False).reset_index(drop=True)


def compare_per_class_runs(
    baseline_df: pd.DataFrame,
    reduced_df: pd.DataFrame,
    class_column: str = "class_name",
    value_column: str = "dice",
) -> pd.DataFrame:
    merged = baseline_df[[class_column, value_column]].rename(
        columns={value_column: "baseline_dice"}
    ).merge(
        reduced_df[[class_column, value_column]].rename(columns={value_column: "reduced_dice"}),
        on=class_column,
        how="inner",
    )
    merged["dice_delta"] = merged["reduced_dice"] - merged["baseline_dice"]
    return merged.sort_values("dice_delta", ascending=False).reset_index(drop=True)
