# engine.py
import os, time, csv
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from torch import amp
from tqdm import tqdm
import numpy as np
import mlflow
import math

import matplotlib.pyplot as plt
import time

from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
import seaborn as sns

# Seaborn custom theme
sns.set_theme(
    style="darkgrid",           
    rc={
        "figure.facecolor": "#0d1b2a",
        "axes.facecolor":   "#0d1b2a",
        "axes.edgecolor":   "#cccccc",
        "grid.color":       "#2a3f5f",
        "axes.labelcolor":  "#ffffff",
        "text.color":       "#ffffff",
        "xtick.color":      "#ffffff",
        "ytick.color":      "#ffffff",
    },
    palette="deep"               
)


def accuracy_topk(output: torch.Tensor, target: torch.Tensor, topk=(1,5)):
    """
    Return counts for each requested top-k.
    e.g. topk=(1,2,3) -> returns [count_top1, count_top2, count_top3]
    """
    maxk = max(topk)
    # get top maxk predictions for each sample
    _, pred = output.topk(maxk, 1, True, True)  # shape: (batch_size, maxk)
    pred = pred.t()  # shape: (maxk, batch_size)
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # (maxk, batch_size)
    res = []
    for k in topk:
        correct_k = correct[:k].any(dim=0).float().sum().item()
        res.append(correct_k)
    return res  # list of counts per requested k

def _main_logits(outputs):
    """Return the primary logits tensor from various model output formats."""
    # torchvision Inception/GoogLeNet may return (logits, aux)
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    # torchvision Inception returns a named output with .logits in some versions
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs



def train_one_epoch(model: nn.Module, loader, optimizer: Optimizer, criterion: nn.Module,
                    scaler: amp.GradScaler, device: torch.device, device_str: str, scheduler: OneCycleLR = None,
                    topk_max: int = 20):
    """Train the model for one epoch and return aggregated statistics and diagnostics."""
    model.train()
    running_loss = 0.0
    running_top_counts = np.zeros(topk_max, dtype=np.float64)  # counts for top1..topK
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
            outputs = _main_logits(outputs)
            loss = criterion(outputs, targets)

        # scale and backward
        scaler.scale(loss).backward()
        try:
            scaler.unscale_(optimizer)
        except Exception:
            pass

        # compute gradient norm (L2) and parameter norm (L2)
        grad_norm_sq = 0.0
        param_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                grad_norm_sq += float(g.norm(2).item() ** 2)
            param_norm_sq += float(p.data.norm(2).item() ** 2)

        last_grad_norm = grad_norm_sq ** 0.5
        last_param_norm = param_norm_sq ** 0.5

        scaler.step(optimizer)
        # call scheduler only if optimizer actually stepped (i.e., grads were finite)
        if scheduler is not None and math.isfinite(last_grad_norm):
            scheduler.step()

        last_lr = optimizer.param_groups[0]['lr'] if len(optimizer.param_groups) > 0 else last_lr
        scaler.update()

        running_loss += loss.item() * images.size(0)
        # get counts for top1..topK
        topk_tuple = tuple(range(1, topk_max + 1))
        t_counts = accuracy_topk(outputs, targets, topk=topk_tuple)
        running_top_counts += np.array(t_counts, dtype=np.float64)
        total += images.size(0)

    epoch_time = time.time() - start_time
    images_per_sec = total / epoch_time if epoch_time > 0 else 0.0

    avg_loss = running_loss / total if total > 0 else float('nan')
    accs = (running_top_counts / total).tolist() if total > 0 else [0.0] * topk_max

    # return avg_loss, list of accs top1..topK, and the other stats
    return avg_loss, accs, float(last_lr), float(last_grad_norm), float(last_param_norm), float(epoch_time), float(images_per_sec)



def validate(model: nn.Module, loader, criterion: nn.Module, device: torch.device,
             device_str: str, topk_max: int = 20, collect: bool = False):
    """Evaluate the model on a dataset and optionally collect predictions and scores."""
    model.eval()
    val_loss = 0.0
    val_top_counts = np.zeros(topk_max, dtype=np.float64)
    vtotal = 0

    start_time = time.time()
    if collect:
        y_true, y_score = [], []
    else:
        y_true, y_score = None, None


    with torch.no_grad():
        for images, targets in tqdm(loader, desc="val"):
            images, targets = images.to(device), targets.to(device)
            with amp.autocast(device_type=device_str):
                outputs = model(images)
                outputs = _main_logits(outputs)
                loss = criterion(outputs, targets)

            val_loss += loss.item() * images.size(0)
            topk_tuple = tuple(range(1, topk_max + 1))
            t_counts = accuracy_topk(outputs, targets, topk=topk_tuple)
            val_top_counts += np.array(t_counts, dtype=np.float64)
            vtotal += images.size(0)

            if collect:
                probs = torch.softmax(outputs, dim=1)
                y_true.append(targets.detach().cpu().numpy())
                y_score.append(probs.detach().cpu().numpy())

    epoch_time = time.time() - start_time
    images_per_sec = vtotal / epoch_time if epoch_time > 0 else 0.0

    val_loss_avg = val_loss / vtotal if vtotal > 0 else float('nan')
    val_accs = (val_top_counts / vtotal).tolist() if vtotal > 0 else [0.0] * topk_max

    if collect:
        return val_loss_avg, val_accs, float(epoch_time), float(images_per_sec), \
               np.concatenate(y_true), np.concatenate(y_score)
    return val_loss_avg, val_accs, float(epoch_time), float(images_per_sec)



def save_checkpoint(path: str, epoch: int, model: nn.Module, optimizer: Optimizer, classes):
    """Save training state including model and optimizer to the given path."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "classes": classes
    }
    torch.save(state, path)

def append_csv(csv_path: str, row, topk_max: int = 20):
    """
    Write a header if file doesn't exist. Header:
    epoch, train_loss, train_acc1..train_accK,
    val_loss, val_acc1..val_accK,
    train_lr, train_grad_norm, train_param_norm,
    train_time_sec, train_images_per_sec,
    val_time_sec, val_images_per_sec
    """
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            headers = (
                ["epoch", "train_loss"]
                + [f"train_acc{k}" for k in range(1, topk_max + 1)]
                + ["val_loss"]
                + [f"val_acc{k}" for k in range(1, topk_max + 1)]
                + [
                    "train_lr",
                    "train_grad_norm",
                    "train_param_norm",
                    "train_time_sec",
                    "train_images_per_sec",
                    "val_time_sec",
                    "val_images_per_sec",
                ]
            )
            writer.writerow(headers)
        writer.writerow(row)


def _collect_predictions(model, dataloader, device, max_store_batches=0):
    """Run inference over dataloader and collect y_true, y_pred, y_score and optionally a small pool of images."""
    model.eval()
    y_true, y_pred, y_score = [], [], []
    images_pool = []

    # infer device type for autocast
    device_str = "cuda" if device.type == "cuda" else "cpu"

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)

            # mixed-precision inference
            with amp.autocast(device_type=device_str):
                logits = model(images)
                logits = _main_logits(logits)
                probs = torch.nn.functional.softmax(logits, dim=1)

            preds = torch.argmax(probs, dim=1)

            y_true.append(targets.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
            y_score.append(probs.cpu().numpy())

            if len(images_pool) < max_store_batches:
                images_pool.append(images.detach().cpu())

    return (
        np.concatenate(y_true),
        np.concatenate(y_pred),
        np.concatenate(y_score),
        images_pool,
    )


def binarize_y(y_true, n_classes):
    """Convert integer labels to one-hot encoded array with n_classes columns."""
    try:
        y_true_b = label_binarize(y_true, classes=list(range(n_classes)))
    except Exception:
        y_true_b = np.eye(n_classes, dtype=int)[y_true]
    if y_true_b.ndim == 1:
        y_true_b = y_true_b.reshape(-1, 1)
    if y_true_b.shape[1] != n_classes:
        y_true_b = np.eye(n_classes, dtype=int)[y_true]
    return y_true_b


def _to_prob_matrix(y_score):
    """
    Return a probability matrix shape (n_samples, n_classes).
    - 2D array: if rows sum approximately to 1 they are treated as probabilities and normalized,
      otherwise treated as logits and converted with softmax.
    - 1D array: treat as positive-class probabilities for binary and convert to (n,2).
    This function also enforces finite values, clips extreme probabilities and ensures
    each row sums exactly to 1.
    """
    y_score = np.asarray(y_score)

    # basic conversion / softmax decision
    if y_score.ndim == 2:
        row_sums = np.sum(y_score, axis=1)
        # treat as probabilities if rows already close to 1 (tolerance kept small)
        if np.allclose(row_sums, 1.0, atol=1e-3):
            probs = y_score.astype(np.float64)
        else:
            logits = y_score.astype(np.float64)
            logits = logits - np.max(logits, axis=1, keepdims=True)
            e = np.exp(logits)
            probs = e / np.sum(e, axis=1, keepdims=True)
    elif y_score.ndim == 1:
        p = y_score.astype(float)
        if np.any((p < 0) | (p > 1)):
            raise ValueError("1-D y_score must be probabilities in [0, 1]")
        probs = np.vstack([1.0 - p, p]).T
    else:
        raise ValueError(f"Unsupported y_score shape: {y_score.shape}")

    # ensure 2-D float64
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        probs = np.squeeze(probs)
        if probs.ndim != 2:
            raise ValueError(f"y_score converted to unexpected shape {probs.shape}; expected (n_samples, n_classes)")

    # handle NaN / Inf by replacing invalid entries with zero, then fix zero-sum rows
    finite_mask = np.isfinite(probs)
    if not finite_mask.all():
        probs = np.where(finite_mask, probs, 0.0)

    # normalize rows to sum 1, replacing zero-sum rows with uniform distribution
    row_sums = probs.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).flatten()
    if zero_rows.any():
        probs[zero_rows] = 1.0 / probs.shape[1]
        row_sums = probs.sum(axis=1, keepdims=True)

    # avoid division by zero and normalize
    row_sums[row_sums == 0] = 1.0
    probs = probs / row_sums

    # clip to safe range for log loss / auc computations and renormalize
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0 - eps)
    probs = probs / probs.sum(axis=1, keepdims=True)

    return probs



def _probs_to_pred(probs, binary_threshold=None):
    """Return integer predicted labels from probability matrix.
    - If 2 classes and binary_threshold is not None use probs[:,1] >= threshold,
      otherwise use argmax across columns.
    """
    probs = np.asarray(probs)
    if probs.ndim != 2:
        raise ValueError("_probs_to_pred expects a 2-D probability matrix")
    n_classes = probs.shape[1]
    if n_classes == 2 and binary_threshold is not None:
        return (probs[:, 1] >= float(binary_threshold)).astype(int)
    return np.argmax(probs, axis=1).astype(int)


def compute_metrics(y_true, y_score, n_bins_calibration=15):
    """Compute classification and probabilistic metrics given true labels and scores."""
    # Normalize/convert scores to probability matrix
    probs = _to_prob_matrix(y_score)  # must return shape (n_samples, n_classes)
    n_classes = int(probs.shape[1])

    # Derive predictions deterministically from probabilities
    y_pred = _probs_to_pred(probs)

    # classification aggregates (per-class)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )

    # convert per-class arrays to lists
    precision_list = [float(x) for x in precision.tolist()]
    recall_list = [float(x) for x in recall.tolist()]
    f1_list = [float(x) for x in f1.tolist()]
    support_list = [int(x) for x in support.tolist()]

    # macro / micro for precision/recall/f1
    macro_prec = float(np.mean(precision)) if precision.size > 0 else float('nan')
    macro_rec = float(np.mean(recall)) if recall.size > 0 else float('nan')
    macro_f1 = float(np.mean(f1)) if f1.size > 0 else float('nan')
    micro_prec, micro_rec, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    micro_prec = float(micro_prec); micro_rec = float(micro_rec); micro_f1 = float(micro_f1)

    # per-class accuracy from confusion matrix diagonal / support
    try:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
        cm_diag = cm.diagonal().astype(float)
        support_arr = cm.sum(axis=1).astype(float)
        # avoid division by zero: where support == 0 set np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class_acc = np.where(support_arr == 0, np.nan, cm_diag / support_arr)
        per_class_acc_list = [float(x) if not np.isnan(x) else float('nan') for x in per_class_acc.tolist()]
    except Exception:
        per_class_acc_list = [float('nan')] * n_classes

    # macro and micro accuracy
    # macro_accuracy: mean of per-class accuracies ignoring nan entries
    try:
        macro_accuracy = float(np.nanmean(per_class_acc)) if per_class_acc.size > 0 else float('nan')
    except Exception:
        macro_accuracy = float('nan')
    try:
        micro_accuracy = float(accuracy_score(y_true, y_pred))
    except Exception:
        micro_accuracy = float('nan')

    # probabilistic metrics: use probs (not raw logits)
    try:
        ll = float(log_loss(y_true, probs, labels=list(range(n_classes))))
    except Exception:
        ll = float('nan')

    try:
        y_true_b = binarize_y(y_true, n_classes)
        per_class_roc_auc = roc_auc_score(y_true_b, probs, average=None, multi_class='ovr')
        per_class_pr_auc = average_precision_score(y_true_b, probs, average=None)
        roc_auc = float(roc_auc_score(y_true_b, probs, average='macro', multi_class='ovr'))
        pr_auc = float(average_precision_score(y_true_b, probs, average='macro'))
        per_class_roc_auc_list = [float(x) for x in per_class_roc_auc.tolist()]
        per_class_pr_auc_list = [float(x) for x in per_class_pr_auc.tolist()]
    except Exception:
        per_class_roc_auc_list = [float('nan')] * n_classes
        per_class_pr_auc_list = [float('nan')] * n_classes
        roc_auc = float('nan')
        pr_auc = float('nan')

    try:
        brier = float(np.mean([brier_score_loss((np.asarray(y_true) == c).astype(int), probs[:, c]) for c in range(n_classes)]))
    except Exception:
        brier = float('nan')

    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = float('nan')
    try:
        kappa = float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        kappa = float('nan')

    # ECE / MCE
    def compute_ece(probs_local, labels, bins=n_bins_calibration):
        """Compute expected and maximum calibration error given probabilities and labels."""
        conf = np.max(probs_local, axis=1)
        preds = np.argmax(probs_local, axis=1)
        correct = (preds == labels).astype(float)
        edges = np.linspace(0.0, 1.0, bins + 1)
        ece_val = 0.0
        mce_val = 0.0
        for i in range(bins):
            low, high = edges[i], edges[i + 1]
            mask = (conf > low) & (conf <= high) if i < bins - 1 else (conf >= low) & (conf <= high)
            if mask.sum() == 0:
                continue
            acc = correct[mask].mean()
            avg_conf = conf[mask].mean()
            gap = abs(acc - avg_conf)
            ece_val += (mask.sum() / len(conf)) * gap
            mce_val = max(mce_val, gap)
        return float(ece_val), float(mce_val)

    try:
        ece, mce = compute_ece(probs, np.asarray(y_true), n_bins_calibration)
    except Exception:
        ece, mce = float('nan'), float('nan')

    # Final dictionary: arrays are turned into lists

    balanced_supports = len(set(support_list)) == 1
    is_single_label = (np.asarray(y_true).ndim == 1 and np.asarray(y_pred).ndim == 1)

    # base fields
    out = dict(
        per_class_accuracy=per_class_acc_list,
        per_class_precision=precision_list,
        per_class_recall=recall_list,
        per_class_f1=f1_list,
        per_class_support=support_list,
        per_class_roc_auc=per_class_roc_auc_list,
        per_class_pr_auc=per_class_pr_auc_list,
        log_loss=ll,
        brier=brier,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        mcc=mcc,
        kappa=kappa,
        ece=ece,
        mce=mce,
    )

    # accuracy keys: collapse when supports are equal
    if balanced_supports:
        out['accuracy'] = macro_accuracy
    else:
        out['macro_accuracy'] = macro_accuracy
        out['micro_accuracy'] = micro_accuracy

    # precision/recall/f1 keys: collapse in single-label multi-class
    if is_single_label:
        out['precision'] = macro_prec
        out['recall'] = macro_rec
        out['f1'] = macro_f1
    else:
        out['macro_prec'] = macro_prec
        out['macro_rec'] = macro_rec
        out['macro_f1'] = macro_f1
        out['micro_prec'] = micro_prec
        out['micro_rec'] = micro_rec
        out['micro_f1'] = micro_f1

    return out


def _save_per_class_csv(path, classes, metrics, topk_rates=None):
    """Save per-class metrics and optional top-k rates into a CSV file."""
    headers = ["label", "precision", "recall", "f1", "support", "roc_auc_ovr", "pr_auc_ovr"]
    if topk_rates is not None:
        headers += [f"top{topk_rates['k']}_rate"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i, cls in enumerate(classes):
            row = [
                str(cls),
                float(metrics['precision'][i]),
                float(metrics['recall'][i]),
                float(metrics['f1'][i]),
                int(metrics['support'][i]),
                (None if metrics['per_class_roc_auc'] is None else float(metrics['per_class_roc_auc'][i])),
                (None if metrics['per_class_pr_auc'] is None else float(metrics['per_class_pr_auc'][i])),
            ]
            if topk_rates is not None:
                row.append(None if np.isnan(topk_rates['rates'][i]) else float(topk_rates['rates'][i]))
            writer.writerow(row)

def _select_top_bottom(arr, k):
    """Return set of indices for top-k and bottom-k (NaNs treated as -inf so they appear in bottom)."""
    if arr is None:
        return set()
    arr = np.asarray(arr)
    if arr.size == 0:
        return set()
    ranked = np.argsort(np.nan_to_num(arr, nan=-np.inf))
    if len(ranked) <= 2 * k:
        return set(ranked.tolist())
    bottom = ranked[:k]
    top = ranked[-k:]
    return set(np.concatenate([top, bottom]))


def plot_roc(y_true_b, y_score, classes, per_class_roc_auc=None,
              roc_path=None, title="ROC", dpi=200, highlight_k=10, save_fig=True):
    """Plot receiver operating characteristic curves for each class and save or display the figure."""
    n_classes = len(classes)
    per_class_roc_auc = (np.asarray(per_class_roc_auc)
                         if per_class_roc_auc is not None
                         else np.full(n_classes, np.nan))

    roc_highlight = _select_top_bottom(per_class_roc_auc, highlight_k)

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        try:
            fpr, tpr, _ = roc_curve(y_true_b[:, i], y_score[:, i])
        except Exception:
            continue
        if i in roc_highlight:
            auc_val = per_class_roc_auc[i]
            auc_label = f"{auc_val:.3f}" if not np.isnan(auc_val) else "nan"
            plt.plot(fpr, tpr, lw=1.6, label=f"{classes[i]} (AUC={auc_label})")
        else:
            plt.plot(fpr, tpr, lw=0.7, alpha=0.9)

    try:
        fpr_micro, tpr_micro, _ = roc_curve(y_true_b.ravel(), y_score.ravel())
        plt.plot(fpr_micro, tpr_micro, linestyle='--', lw=2.0, label="micro (aggregated)")
    except Exception:
        pass

    plt.plot([0, 1], [0, 1], '--', lw=0.8, alpha=0.6)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)

    if n_classes <= 12:
        plt.legend(loc='lower right', fontsize='small', ncol=2)
    else:
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='x-small', frameon=False)

    plt.tight_layout(rect=[0, 0, 0.78, 1])

    if save_fig and roc_path:
        os.makedirs(os.path.dirname(roc_path), exist_ok=True)
        plt.savefig(roc_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_pr(y_true_b, y_score, classes, per_class_pr_auc=None,
             pr_path=None, title="Precision-Recall", dpi=200, highlight_k=10, save_fig=True):
    """Plot precision-recall curves for each class and save or display the figure."""
    n_classes = len(classes)
    per_class_pr_auc = (np.asarray(per_class_pr_auc)
                        if per_class_pr_auc is not None
                        else np.full(n_classes, np.nan))

    pr_highlight = _select_top_bottom(per_class_pr_auc, highlight_k)

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        try:
            prec, rec, _ = precision_recall_curve(y_true_b[:, i], y_score[:, i])
        except Exception:
            continue
        if i in pr_highlight:
            ap_val = per_class_pr_auc[i]
            ap_label = f"{ap_val:.3f}" if not np.isnan(ap_val) else "nan"
            plt.plot(rec, prec, lw=1.6, label=f"{classes[i]} (AP={ap_label})")
        else:
            plt.plot(rec, prec, lw=0.7, alpha=0.9)

    try:
        prec_micro, rec_micro, _ = precision_recall_curve(y_true_b.ravel(), y_score.ravel())
        plt.plot(rec_micro, prec_micro, linestyle='--', lw=2.0, label="micro (aggregated)")
    except Exception:
        pass

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)

    if n_classes <= 12:
        plt.legend(loc='lower left', fontsize='small', ncol=2)
    else:
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='x-small', frameon=False)

    plt.tight_layout(rect=[0, 0, 0.78, 1])

    if save_fig and pr_path:
        os.makedirs(os.path.dirname(pr_path), exist_ok=True)
        plt.savefig(pr_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_calibration(y_score, y_true, figsize = (6, 4),n_bins=20, dpi=200, save_fig=True, calib_path = None):
    """Reliability diagram: mean predicted probability vs fraction of positives."""
    confidences = np.max(y_score, axis=1)
    predictions = np.argmax(y_score, axis=1)
    correctness = (predictions == y_true).astype(int)
    prob_true, prob_pred = calibration_curve(correctness, confidences, n_bins=n_bins, strategy='uniform')

    plt.figure(figsize=figsize)
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1)
    plt.plot([0, 1], [0, 1], '--', color='w')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Reliability diagram')
    plt.tight_layout()

    if save_fig and calib_path:
        os.makedirs(os.path.dirname(calib_path), exist_ok=True)
        plt.savefig(calib_path, dpi=dpi)
    else:
        plt.show()
    plt.close()


def plot_confidence_hist(y_score, figsize = (6, 4), bins=30, dpi=200, save_fig=True, confhist_path = None):
    """ Histogram of predicted max confidence values. """
    confidences = np.max(y_score, axis=1)

    plt.figure(figsize=figsize)
    plt.hist(confidences, bins=bins)
    plt.xlabel('Predicted max confidence')
    plt.ylabel('Count')
    plt.title('Confidence histogram')
    plt.tight_layout()

    if save_fig and confhist_path:
        os.makedirs(os.path.dirname(confhist_path), exist_ok=True)
        plt.savefig(confhist_path, dpi=dpi)
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(cm_norm, classes, title = "Confusion Matrix", cm_path = None, dpi=200, save_fig = True):
    """Render a confusion matrix heatmap and save or display the figure."""
    plt.figure(figsize=(8,8), dpi=dpi)
    im = plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=90, fontsize=6)
    plt.yticks(ticks, classes, fontsize=6)
    plt.ylabel('True label'); plt.xlabel('Predicted label')
    plt.tight_layout()
    if save_fig:
        plt.savefig(cm_path, dpi=dpi)
    else:
        plt.show()
    plt.close()
        
def plot_grid(idxs, sample_images, sample_true, sample_pred, sample_score, path, title, classes,  max_images=25, dpi=120, save_fig=True):
    """Create a grid of sample images annotated with true and predicted labels and probabilities."""
    if not idxs:
        return None

    take = idxs[:max_images]
    n = len(take)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    for ax in axes.flatten():
        ax.axis('off')

    for k, idx in enumerate(take):
        r = k // cols; c = k % cols
        ax = axes[r, c]
        img = sample_images[idx]
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std = np.array([0.229, 0.224, 0.225])[:, None, None]
        img_np = img.numpy()
        img_np = (img_np * std) + mean
        img_np = np.clip(img_np, 0, 1)
        img_np = np.transpose(img_np, (1, 2, 0))
        ax.imshow(img_np)
        tlabel = classes[sample_true[idx]]
        plabel = classes[sample_pred[idx]]
        conf = float(np.max(sample_score[idx]))
        ax.set_title(f"T:{tlabel}\nP:{plabel} ({conf:.2f})", fontsize=6)

    plt.suptitle(f"{title}", fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    if save_fig:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return path

    plt.show()
    plt.close(fig)
    return None


def _mlflow_log(run_metrics: dict, artifact_paths: dict, epoch: int):
    """Log scalar metrics and artifacts to MLflow for the provided epoch."""
    # scalars
    for k, v in run_metrics.items():
        if v is None:
            continue
        mlflow.log_metric(f"val/{k}", float(v), step=epoch)
    # artifacts: artifact_paths is {category: path or list}
    for apath in artifact_paths.values():
        if apath is None:
            continue
        if isinstance(apath, (list, tuple)):
            for p in apath:
                if p:
                    mlflow.log_artifact(p)
        else:
            mlflow.log_artifact(apath)
            
            

# ---------- Main evaluate_and_log (orchestrator) ----------

def evaluate_and_log(model, dataloader, device, epoch, classes, run_name, mlflow_enabled=True,
                     max_misclassified_images=25, n_bins_calibration=15, topk_for_visual=5,
                     base_artifact_dir="artifacts", dpi=200, y_true=None, y_score=None):
    """
    Evaluate the model over a dataloader, save metrics and artifacts and optionally log results to MLflow.
    Saves plots, CSVs and prediction archives for the given run and epoch.
    Returns a summary dictionary with aggregate metrics and artifact paths.
    """
    start = time.time()
    os.makedirs(base_artifact_dir, exist_ok=True)
    subdirs = ["plots", "inspection", "predictions", "per_class_metrics", "confusion_matrix"]
    
    for d in subdirs:
        base = os.path.join(base_artifact_dir, d)
        run = base if d == "predictions" else os.path.join(base, run_name)
        os.makedirs(run, exist_ok=True) 


    # collect predictions
    if y_true is None or y_score is None:
        y_true, y_pred, y_score, images_pool = _collect_predictions(model, dataloader, device)
    else:
        # Convert provided y_score to a proper probability matrix and compute predictions
        probs = _to_prob_matrix(y_score)
        y_pred = _probs_to_pred(probs)
        y_score = probs
        images_pool = []

    # metrics
    n_classes = len(classes)
    metrics = compute_metrics(y_true, y_score, n_bins_calibration)

    #confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    cm_csv_path = os.path.join(base_artifact_dir, "confusion_matrix", run_name, f"epoch{epoch}.csv")
    # write confusion matrix to CSV (rows = true labels, columns = predicted labels)
    with open(cm_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["true_label"] + [str(c) for c in classes]
        writer.writerow(header)
        for i, row in enumerate(cm):
            writer.writerow([str(classes[i])] + row.tolist())
    

    
    # per-class CSV
    per_class_path = os.path.join(base_artifact_dir, "per_class_metrics",run_name, f"epoch{epoch}.csv")
    # optional top-k rates
    topk_rates = None
    try:
        k = topk_for_visual
        topk_preds = np.argsort(y_score, axis=1)[:, ::-1][:, :k]
        rates = []
        for c in range(n_classes):
            idxs = np.where(y_true == c)[0]
            if len(idxs) == 0:
                rates.append(np.nan)
            else:
                hits = np.sum([1 if c in topk_preds[i] else 0 for i in idxs])
                rates.append(hits / len(idxs))
        topk_rates = dict(k=k, rates=rates)
    except Exception:
        topk_rates = dict(k=topk_for_visual, rates=[np.nan]*n_classes)

    _save_per_class_csv(per_class_path, classes, metrics, topk_rates=topk_rates)

    # predictions NPZ (save every 5 epoch)
    preds_npz_path = None
    if epoch %5 == 0:
        preds_npz_path = os.path.join(base_artifact_dir, "predictions", f"{run_name}.npz")
        np.savez_compressed(preds_npz_path ,y_true=y_true.astype(np.int32),
                            y_pred=y_pred.astype(np.int32),
                            y_score=y_score.astype(np.float32),
                            classes=np.array(list(classes)))

    # plots (ROC/PR, calibration, CM)
    plots_dir = os.path.join(base_artifact_dir, "plots")
    roc_plot_path = os.path.join(plots_dir, run_name, f"roc_epoch{epoch}.png")
    pr_plot_path = os.path.join(plots_dir, run_name, f"pr_epoch{epoch}.png")
    calib_plot_path = os.path.join(plots_dir, run_name, f"reliability_epoch{epoch}.png")
    confhist_plot_path = os.path.join(plots_dir, run_name, f"confidence_hist_epoch{epoch}.png")
    cm_plot_path = os.path.join(plots_dir, run_name, f"cm_epoch{epoch}_cm.png")

    y_true_b = binarize_y(y_true, n_classes)
    if len(classes) <= 30: 
        plot_roc(
            y_true_b, y_score, classes, metrics['per_class_roc_auc'],roc_path=roc_plot_path,
            title=f"ROC - {run_name} epoch {epoch}", dpi=dpi, highlight_k=10
        )
        plot_pr(
            y_true_b, y_score, classes, metrics['per_class_pr_auc'],pr_path=pr_plot_path,
            title=f"Precision-Recall - {run_name} epoch {epoch}",dpi=dpi, highlight_k=10
        )

    else:
        roc_plot_path = pr_plot_path = None
    try:
        plot_calibration(y_score, y_true, n_bins=n_bins_calibration, dpi=dpi, calib_path=calib_plot_path)
        plot_confidence_hist(y_score, bins=30, dpi=dpi, confhist_path = confhist_plot_path)
    except Exception:
        calib_plot_path = None
        confhist_plot_path = None
    try:
        plot_confusion_matrix(cm, classes, f"Normalized Confusion Matrix — {run_name} epoch {epoch}", cm_plot_path, dpi=dpi)
    except Exception:
        cm_plot_path = None

    # optionally create small inspection grids (keeps bounded memory)
    misclassified_grid_path = None
    lowconf_grid_path = None
    try:
        # small sample pass to collect inspectable images (bounded)
        sample_images, sample_true, sample_pred, sample_score = [], [], [], []
        max_samples_to_collect = max_misclassified_images * 4
        device_str = "cuda" if device.type == "cuda" else "cpu"

        with torch.no_grad():
            collected = 0
            for images, targets in dataloader:
                images = images.to(device)

                # mixed-precision inference for inspection as well
                with amp.autocast(device_type=device_str):
                    logits = model(images)
                    logits = _main_logits(logits)
                    probs = torch.nn.functional.softmax(logits, dim=1)

                preds = torch.argmax(probs, dim=1)

                for i in range(images.size(0)):
                    sample_images.append(images[i].cpu())
                    sample_true.append(int(targets[i].item()))
                    sample_pred.append(int(preds[i].item()))
                    sample_score.append(probs[i].cpu().numpy())
                    collected += 1
                    if collected >= max_samples_to_collect:
                        break
                if collected >= max_samples_to_collect:
                    break

        mis_idxs = [i for i in range(len(sample_images)) if sample_pred[i] != sample_true[i]]
        lowconf_idxs = [i for i in range(len(sample_images)) if sample_pred[i] == sample_true[i] and np.max(sample_score[i]) < 0.6]
        inspection_path = os.path.join(base_artifact_dir, "inspection",run_name)

        misclassified_grid_path = plot_grid(
            mis_idxs, sample_images, sample_true, sample_pred, sample_score, os.path.join(inspection_path, f"misclassified_epoch{epoch}.png"),
            f"Misclassified - {run_name} epoch {epoch}", classes, dpi=dpi
        )
        lowconf_grid_path = plot_grid(
            lowconf_idxs,sample_images,sample_true,sample_pred, sample_score, os.path.join(inspection_path, f"lowconf_correct_epoch{epoch}.png"),
            f"Low-confidence Correct - {run_name} epoch {epoch}", classes, dpi=dpi
        )

    except Exception:
        misclassified_grid_path = None; lowconf_grid_path = None

    # MLflow logging (if enabled)
    if mlflow_enabled:
        # determine whether compute_metrics collapsed keys
        balanced_supports = 'accuracy' in metrics
        single_label = 'precision' in metrics

        # always include these core metrics
        run_metrics = {
            "roc_auc": metrics.get('roc_auc'),
            "pr_auc": metrics.get('pr_auc'),
            "brier": metrics.get('brier'),
            "log_loss": metrics.get('log_loss'),
            "matthews_corrcoef": metrics.get('mcc'),
            "cohen_kappa": metrics.get('kappa'),
            "ece": metrics.get('ece'),
            "mce": metrics.get('mce'),
            "epoch_time_sec": time.time() - start
        }

        # conditional accuracy fields
        if balanced_supports:
            run_metrics["accuracy"] = metrics.get('accuracy')
        else:
            run_metrics["macro_accuracy"] = metrics.get('macro_accuracy')
            run_metrics["micro_accuracy"] = metrics.get('micro_accuracy')

        # conditional precision/recall/f1 fields
        if single_label:
            run_metrics["precision"] = metrics.get('precision')
            run_metrics["recall"] = metrics.get('recall')
            run_metrics["f1"] = metrics.get('f1')
        else:
            run_metrics["macro_precision"] = metrics.get('macro_prec')
            run_metrics["macro_recall"] = metrics.get('macro_rec')
            run_metrics["macro_f1"] = metrics.get('macro_f1')
            run_metrics["micro_precision"] = metrics.get('micro_prec')
            run_metrics["micro_recall"] = metrics.get('micro_rec')
            run_metrics["micro_f1"] = metrics.get('micro_f1')

        artifact_paths = dict(
            per_class=per_class_path,
            cm=cm_csv_path,
            roc=roc_plot_path,
            pr=pr_plot_path,
            calib=calib_plot_path,
            confhist=confhist_plot_path,
            cm_plot=cm_plot_path,
            misclassified=misclassified_grid_path,
            lowconf=lowconf_grid_path,
            predictions=preds_npz_path
        )
        _mlflow_log(run_metrics, artifact_paths, epoch)

    balanced_supports = 'accuracy' in metrics
    single_label = 'precision' in metrics

    summary = dict(
        roc_auc=metrics.get('roc_auc'),
        pr_auc=metrics.get('pr_auc'),
        brier=metrics.get('brier'),
        log_loss=metrics.get('log_loss'),
        matthews_corrcoef=metrics.get('mcc'),
        cohen_kappa=metrics.get('kappa'),
        ece=metrics.get('ece'),
        mce=metrics.get('mce'),
        per_class_metrics_path=per_class_path,
        confusion_matrix_csv_path=cm_csv_path,
        predictions_npz_path=preds_npz_path,
        roc_plot_path=roc_plot_path,
        pr_plot_path=pr_plot_path,
        calibration_plot_path=calib_plot_path,
        confidence_histogram_path=confhist_plot_path,
        cm_plot_path=cm_plot_path,
        misclassified_grid_path=misclassified_grid_path,
        lowconf_grid_path=lowconf_grid_path,
        epoch_time=float(time.time() - start),
    )

    if balanced_supports:
        summary['accuracy'] = metrics.get('accuracy')
    else:
        summary['macro_accuracy'] = metrics.get('macro_accuracy')
        summary['micro_accuracy'] = metrics.get('micro_accuracy')

    if single_label:
        summary['precision'] = metrics.get('precision')
        summary['recall'] = metrics.get('recall')
        summary['f1'] = metrics.get('f1')
    else:
        summary['macro_precision'] = metrics.get('macro_prec')
        summary['macro_recall'] = metrics.get('macro_rec')
        summary['macro_f1'] = metrics.get('macro_f1')
        summary['micro_precision'] = metrics.get('micro_prec')
        summary['micro_recall'] = metrics.get('micro_rec')
        summary['micro_f1'] = metrics.get('micro_f1')

    return summary
