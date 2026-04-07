from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix

from experiment_utils import (
    append_row_to_csv,
    build_run_paths,
    elapsed_seconds,
    save_history,
    save_json,
    save_per_class_metrics,
    timer,
)


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    best_score: float = -math.inf
    epochs_without_improvement: int = 0

    def update(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.epochs_without_improvement = 0
            return False
        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loss():
    return nn.CrossEntropyLoss()


def _compute_per_class_metrics(confmat: torch.Tensor, class_names: Dict[int, str], background_index: int = 0):
    confmat = confmat.float()
    tp = torch.diag(confmat)
    fp = confmat.sum(dim=0) - tp
    fn = confmat.sum(dim=1) - tp
    denom = 2 * tp + fp + fn
    dice = torch.where(denom > 0, (2 * tp) / denom, torch.zeros_like(tp))
    iou_denom = tp + fp + fn
    iou = torch.where(iou_denom > 0, tp / iou_denom, torch.zeros_like(tp))

    per_class_rows = []
    for class_idx in range(confmat.shape[0]):
        per_class_rows.append(
            {
                "class_index": int(class_idx),
                "class_name": class_names.get(class_idx, f"class_{class_idx}"),
                "dice": float(dice[class_idx].item()),
                "iou": float(iou[class_idx].item()),
                "support": float(confmat[class_idx].sum().item()),
                "is_background": bool(class_idx == background_index),
            }
        )

    foreground_indices = [idx for idx in range(confmat.shape[0]) if idx != background_index]
    mdice_ex_bg = float(dice[foreground_indices].mean().item()) if foreground_indices else 0.0
    miou_ex_bg = float(iou[foreground_indices].mean().item()) if foreground_indices else 0.0
    return per_class_rows, mdice_ex_bg, miou_ex_bg


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_batches = 0
    use_amp = scaler is not None and device.type == "cuda"

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = loss_fn(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().item())
        total_batches += 1

    return {"train_loss": total_loss / max(total_batches, 1)}


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
    class_names: Dict[int, str],
    background_index: int = 0,
    max_batches: Optional[int] = None,
) -> Dict:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    pixel_accuracy = MulticlassAccuracy(
        num_classes=num_classes,
        average="micro",
        multidim_average="global",
        ignore_index=background_index,
    ).to(device)
    confusion = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        loss = loss_fn(logits, masks)
        preds = torch.argmax(logits, dim=1)

        total_loss += float(loss.item())
        total_batches += 1
        pixel_accuracy.update(preds, masks)
        confusion.update(preds, masks)

    confmat = confusion.compute().detach().cpu()
    per_class_rows, mdice_ex_bg, miou_ex_bg = _compute_per_class_metrics(
        confmat=confmat,
        class_names=class_names,
        background_index=background_index,
    )
    return {
        "val_loss": total_loss / max(total_batches, 1),
        "val_pixel_acc_ex_bg": float(pixel_accuracy.compute().detach().cpu().item()),
        "val_mdice_ex_bg": mdice_ex_bg,
        "val_miou_ex_bg": miou_ex_bg,
        "per_class_metrics": per_class_rows,
        "confusion_matrix": confmat.numpy(),
    }


def save_checkpoint(path: str | Path, payload: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def run_experiment(
    config: Dict,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: Dict[int, str],
    output_root: str | Path,
) -> Dict:
    if config.get("model_name") == "upernet_swin_tiny":
        target_size = int(config["image_size"])
        current_size = None
        try:
            current_size = int(model.encoder.model.patch_embed.img_size[0])
        except Exception:
            current_size = None

        if current_size != target_size:
            from model_factory import build_model

            model = build_model(
                config["model_name"],
                num_classes=config["num_classes"],
                image_size=target_size,
            )

    device = get_device()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    loss_fn = build_loss()
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    run_paths = build_run_paths(output_root, config)

    if run_paths["checkpoint"].exists() and not config.get("force_rerun", False):
        return {
            "status": "skipped_existing",
            "run_name": run_paths["run_name"],
            "checkpoint_path": str(run_paths["checkpoint"]),
        }

    save_json(run_paths["config_json"], config)

    stopper = None
    if config.get("early_stopping_patience") is not None:
        stopper = EarlyStopping(
            patience=config["early_stopping_patience"],
            min_delta=config.get("early_stopping_delta", 0.0),
        )

    history: List[Dict] = []
    best_metrics: Optional[Dict] = None
    best_epoch = -1
    best_score = -math.inf
    start_time = timer()

    for epoch in range(1, config["epochs"] + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            max_batches=config.get("max_train_batches"),
        )
        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            num_classes=config["num_classes"],
            class_names=class_names,
            background_index=config.get("background_index", 0),
            max_batches=config.get("max_val_batches"),
        )

        epoch_row = {
            "epoch": epoch,
            **train_metrics,
            "val_loss": val_metrics["val_loss"],
            "val_pixel_acc_ex_bg": val_metrics["val_pixel_acc_ex_bg"],
            "val_mdice_ex_bg": val_metrics["val_mdice_ex_bg"],
            "val_miou_ex_bg": val_metrics["val_miou_ex_bg"],
        }
        history.append(epoch_row)

        current_score = val_metrics["val_mdice_ex_bg"]
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            best_metrics = val_metrics
            save_checkpoint(
                run_paths["checkpoint"],
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "best_metrics": {
                        "val_loss": val_metrics["val_loss"],
                        "val_pixel_acc_ex_bg": val_metrics["val_pixel_acc_ex_bg"],
                        "val_mdice_ex_bg": val_metrics["val_mdice_ex_bg"],
                        "val_miou_ex_bg": val_metrics["val_miou_ex_bg"],
                    },
                },
            )
            save_per_class_metrics(best_metrics["per_class_metrics"], run_paths["per_class_csv"])

        if stopper is not None and stopper.update(current_score):
            break

    total_seconds = elapsed_seconds(start_time)
    save_history(history, run_paths["history_csv"])

    if best_metrics is None:
        raise RuntimeError("Experiment finished without validation metrics.")

    summary_row = {
        "run_name": run_paths["run_name"],
        "status": "completed",
        "model_name": config["model_name"],
        "image_size": config["image_size"],
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "augmentation": config["augmentation"],
        "run_mode": config["run_mode"],
        "reduced_label_tag": config.get("reduced_label_tag", "full_labels"),
        "num_classes": config["num_classes"],
        "epochs": config["epochs"],
        "best_epoch": best_epoch,
        "best_val_mdice_ex_bg": best_metrics["val_mdice_ex_bg"],
        "best_val_pixel_acc_ex_bg": best_metrics["val_pixel_acc_ex_bg"],
        "best_val_miou_ex_bg": best_metrics["val_miou_ex_bg"],
        "seconds": total_seconds,
        "checkpoint_path": str(run_paths["checkpoint"]),
        "history_csv": str(run_paths["history_csv"]),
        "per_class_csv": str(run_paths["per_class_csv"]),
        "config_json": str(run_paths["config_json"]),
    }
    append_row_to_csv(run_paths["summary_csv"], summary_row)
    return summary_row
