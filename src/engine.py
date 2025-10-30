# engine.py
import csv
import os
from typing import Tuple
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from torch import amp
from tqdm import tqdm
import numpy as np
import mlflow
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import time
import json


def accuracy_topk(output: torch.Tensor, target: torch.Tensor, topk=(1,5)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        res.append(correct[:k].reshape(-1).float().sum(0, keepdim=True).item())
    return res  # list [top1_count, top5_count]

import time
# other imports ...

def train_one_epoch(model: nn.Module, loader, optimizer: Optimizer, criterion: nn.Module,
                    scaler: amp.GradScaler, device: torch.device, device_str: str, scheduler: OneCycleLR = None):
    model.train()
    running_loss = 0.0
    running_top1 = 0
    running_top5 = 0
    total = 0

    start_time = time.time()

    # variables to hold last-seen grad/param norms and lr
    last_grad_norm = 0.0
    last_param_norm = 0.0
    last_lr = optimizer.param_groups[0]['lr'] if len(optimizer.param_groups) > 0 else 0.0

    for images, targets in tqdm(loader, desc="train"):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        with amp.autocast(device_type=device_str):
            outputs = model(images)
            loss = criterion(outputs, targets)

        # scale and backward
        scaler.scale(loss).backward()
        try:
            # GradScaler has method unscale_ in current PyTorch API
            scaler.unscale_(optimizer)
        except Exception:
            # If unscale_ is not available, continue — gradient norms may be scaled (best-effort)
            pass

        # compute gradient norm (L2) and parameter norm (L2) at this point
        grad_norm_sq = 0.0
        param_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                grad_norm_sq += float(g.norm(2).item() ** 2)
            # parameter norm (always available)
            param_norm_sq += float(p.data.norm(2).item() ** 2)

        last_grad_norm = grad_norm_sq ** 0.5
        last_param_norm = param_norm_sq ** 0.5

        scaler.step(optimizer)
        if scheduler is not None:
            scheduler.step()
        last_lr = optimizer.param_groups[0]['lr'] if len(optimizer.param_groups) > 0 else last_lr
        scaler.update()

        running_loss += loss.item() * images.size(0)
        t1, t5 = accuracy_topk(outputs, targets, topk=(1,5))
        running_top1 += t1
        running_top5 += t5
        total += images.size(0)

    epoch_time = time.time() - start_time
    images_per_sec = total / epoch_time if epoch_time > 0 else 0.0

    avg_loss = running_loss / total if total > 0 else float('nan')
    acc1 = running_top1 / total if total > 0 else 0.0
    acc5 = running_top5 / total if total > 0 else 0.0

    # return existing values plus the extra metrics
    return avg_loss, acc1, acc5, float(last_lr), float(last_grad_norm), float(last_param_norm), float(epoch_time), float(images_per_sec)


def validate(model: nn.Module, loader, criterion: nn.Module, device: torch.device, device_str: str):
    model.eval()
    val_loss, val_top1, val_top5, vtotal = 0.0, 0, 0, 0
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="val"):
            images, targets = images.to(device), targets.to(device)
            with amp.autocast(device_type=device_str):
                outputs = model(images)
                loss = criterion(outputs, targets)

            val_loss += loss.item() * images.size(0)
            t1, t5 = accuracy_topk(outputs, targets, topk=(1,5))
            val_top1 += t1
            val_top5 += t5
            vtotal += images.size(0)

    return val_loss / vtotal, val_top1 / vtotal, val_top5 / vtotal

def save_checkpoint(path: str, epoch: int, model: nn.Module, optimizer: Optimizer, classes):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "classes": classes
    }
    torch.save(state, path)

def append_csv(csv_path: str, row):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["epoch","train_loss","train_acc1","train_acc5","val_loss","val_acc1","val_acc5"])
        writer.writerow(row)


def evaluate_and_log(model, dataloader, device, epoch, classes, mlflow_enabled=True):
    """
    Run a full pass to gather probabilities, predictions and targets; compute
    multi-class metrics (including per-class precision/recall/F1, per-class
    ROC-AUC and PR-AUC (OVR)), save a confusion-matrix figure, store a per-class
    JSON artifact and log scalar metrics to MLflow when enabled.

    Returns a dict with scalar summary values and paths to generated artifacts
    when available.
    """
    model.eval()
    n_classes = len(classes)

    # placeholders for artifact paths (defined even if mlflow_enabled is False)
    cm_path = None
    per_class_path = None

    y_true = []
    y_pred = []
    y_score = []  # probabilities
    start = time.time()
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.append(targets.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
            y_score.append(probs.cpu().numpy())

    epoch_time = time.time() - start
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_score = np.concatenate(y_score)  # shape (N, n_classes)

    # Basic per-class & global metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )
    macro_prec = float(np.mean(precision))
    macro_rec = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))
    micro_prec, micro_rec, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )

    # ROC-AUC / PR-AUC: compute per-class (OVR) arrays and macro averages
    per_class_roc_auc = None
    per_class_pr_auc = None
    try:
        # Try to binarize labels; fallback to a NumPy one-hot if necessary
        try:
            y_true_b = label_binarize(y_true, classes=list(range(n_classes)))
        except Exception:
            y_true_b = np.eye(n_classes, dtype=int)[y_true]

        # Ensure shape is (n_samples, n_classes)
        if y_true_b.ndim == 1:
            y_true_b = y_true_b.reshape(-1, 1)
        if y_true_b.shape[1] != n_classes:
            # fallback: construct explicit one-hot
            y_true_b = np.eye(n_classes, dtype=int)[y_true]

        # per-class OVR scores (array of length n_classes)
        per_class_roc_auc = roc_auc_score(y_true_b, y_score, average=None, multi_class='ovr')
        per_class_pr_auc = average_precision_score(y_true_b, y_score, average=None)

        # macro averages (scalars)
        roc_auc = float(roc_auc_score(y_true_b, y_score, average='macro', multi_class='ovr'))
        pr_auc = float(average_precision_score(y_true_b, y_score, average='macro'))
    except Exception:
        roc_auc = float('nan')
        pr_auc = float('nan')

    # Brier score (macro average across classes)
    try:
        brier = float(np.mean([brier_score_loss((y_true == c).astype(int), y_score[:, c]) for c in range(n_classes)]))
    except Exception:
        brier = float('nan')

    # Confusion matrix & save figure
    try:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax, xticks_rotation='vertical', values_format='d')
        plt.title(f'Confusion Matrix (epoch {epoch})')
        cm_path = f'artifacts/confusion_matrix_epoch{epoch}.png'
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close(fig)
    except Exception:
        cm_path = None

    # Log numeric metrics to MLflow (scalars) and save per-class JSON artifact
    if mlflow_enabled:
        # scalar metrics
        mlflow.log_metric("val/macro_precision", float(macro_prec), step=epoch)
        mlflow.log_metric("val/macro_recall", float(macro_rec), step=epoch)
        mlflow.log_metric("val/macro_f1", float(macro_f1), step=epoch)
        mlflow.log_metric("val/micro_precision", float(micro_prec), step=epoch)
        mlflow.log_metric("val/micro_recall", float(micro_rec), step=epoch)
        mlflow.log_metric("val/micro_f1", float(micro_f1), step=epoch)
        mlflow.log_metric("val/roc_auc", float(roc_auc), step=epoch)
        mlflow.log_metric("val/pr_auc", float(pr_auc), step=epoch)
        mlflow.log_metric("val/brier", float(brier), step=epoch)
        mlflow.log_metric("val/epoch_time_sec", float(epoch_time), step=epoch)

        # per-class metrics as JSON artifact
        per_class = {
            "labels": list(classes),
            "precision": [float(x) for x in precision.tolist()],
            "recall":    [float(x) for x in recall.tolist()],
            "f1":        [float(x) for x in f1.tolist()],
            "support":   [int(x) for x in support.tolist()],
            "roc_auc_ovr": ([] if per_class_roc_auc is None else [float(x) for x in np.ravel(per_class_roc_auc).tolist()]),
            "pr_auc_ovr":  ([] if per_class_pr_auc  is None else [float(x) for x in np.ravel(per_class_pr_auc).tolist()]),
        }
        per_class_path = f'artifacts/per_class_metrics_epoch{epoch}.json'
        try:
            with open(per_class_path, 'w') as f:
                json.dump(per_class, f, indent=2)
            mlflow.log_dict(per_class, f"metrics/per_class_metrics_epoch{epoch}.json")
            mlflow.log_artifact(per_class_path, artifact_path="metrics")
        except Exception:
            per_class_path = None

        # log confusion matrix artifact if available
        if cm_path is not None:
            try:
                mlflow.log_artifact(cm_path, artifact_path="plots")
            except Exception:
                pass

    # return a dict for local consumption if needed
    return {
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "micro_precision": float(micro_prec),
        "micro_recall": float(micro_rec),
        "micro_f1": float(micro_f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "confusion_matrix_path": cm_path,
        "per_class_metrics_path": per_class_path,
        "epoch_time": epoch_time,
    }
