#!/usr/bin/env python3
"""
Evaluation script for Food-101 models.

Usage example:
  python eval.py \
    --checkpoint artifacts/checkpoints/best.pth \
    --data-root /path/to/data \
    --out-dir artifacts/eval \
    --batch-size 64 \
    --device cuda
"""
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.calibration import calibration_curve

# ---- Dataset helper for Food-101 test split ----
class Food101TestDataset(Dataset):
    def __init__(self, data_root: str, split="test", transform=None):
        """
        Expects Food-101 layout:
        data_root/food-101/images/<class>/<image>.jpg
        data_root/food-101/meta/test.txt (each line like: 'apple_pie/12345')
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        meta_dir = self.data_root / "food-101" / "meta"
        if not meta_dir.exists():
            raise FileNotFoundError(f"Cannot find meta dir: {meta_dir}")
        split_file = meta_dir / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Cannot find {split}.txt at {split_file}")

        self.entries = []
        with open(split_file, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                # each line is like 'apple_pie/00000001'
                img_rel = f"{line}.jpg"
                img_path = self.data_root / "food-101" / "images" / img_rel
                label = line.split("/")[0]
                self.entries.append((str(img_path), label))

        # build class -> idx mapping
        classes = sorted({label for _, label in self.entries})
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label = self.entries[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[label], path

# ---- Helpers to load classes and model from checkpoint ----
def try_load_classes(checkpoint_path: Path, out_dir: Path):
    # try sidecar classes.json near checkpoint
    cjson = checkpoint_path.parent / "classes.json"
    if cjson.exists():
        try:
            with open(cjson, "r") as fh:
                classes = json.load(fh)
            return classes
        except Exception:
            pass
    # try to inspect checkpoint
    try:
        ck = torch.load(checkpoint_path, map_location="cpu")
        # try several common keys
        for key in ("classes", "class_to_idx", "idx_to_class"):
            if key in ck:
                return ck[key]
        # maybe checkpoint saved 'meta' or 'labels'
        if isinstance(ck, dict):
            for k in ck.keys():
                if 'class' in k.lower() or 'label' in k.lower() or 'idx' in k.lower():
                    return ck[k]
    except Exception:
        pass
    # not found
    return None

def build_model_from_checkpoint(checkpoint_path: Path, device: torch.device, num_classes=None):
    """
    Strategy:
      1. Try to import src.food101.model.build_model() if present.
      2. Try to load a state_dict and require the caller to pass num_classes or infer from checkpoint keys.
      3. If checkpoint contains a full serialized model object under 'model', attempt to use it.
    """
    # attempt to import repo's builder
    try:
        import importlib
        model_mod = importlib.import_module("src.food101.model")
        if hasattr(model_mod, "build_model"):
            builder = getattr(model_mod, "build_model")
            # builder should accept num_classes or pretrained arg; try conservative call
            if num_classes is None:
                # try to guess num_classes from checkpoint classes or raise later
                num_classes = None
            model = builder(num_classes=num_classes) if "num_classes" in builder.__code__.co_varnames else builder()
            model.to(device)
            return model
    except Exception:
        pass

    # fallback: try to load checkpoint and reconstruct
    ck = torch.load(checkpoint_path, map_location="cpu")
    # cases: ck may be a dict with 'state_dict' or 'model_state_dict'
    sd = None
    if isinstance(ck, dict):
        if "state_dict" in ck:
            sd = ck["state_dict"]
        elif "model_state_dict" in ck:
            sd = ck["model_state_dict"]
        elif "model" in ck and isinstance(ck["model"], torch.nn.Module):
            model = ck["model"]
            model.to(device)
            return model
        elif "state_dict" in ck:
            sd = ck["state_dict"]

    # If we have just a state_dict but no builder, try to infer architecture keys (resnet, vit) and create a minimal model
    if sd is not None:
        # guess a common arch (ResNet-ish) by checking key names
        keys = list(sd.keys())
        if any(k.startswith("fc.") for k in keys) or any("layer1" in k for k in keys):
            # create a ResNet50 and replace final layer
            try:
                import torchvision.models as models
                model = models.resnet50(pretrained=False)
                if num_classes is None:
                    # attempt to infer num_classes if final fc has bias/weight shapes in state_dict
                    fc_w = sd.get("fc.weight")
                    if fc_w is not None:
                        num_classes = fc_w.shape[0]
                if num_classes is None:
                    raise RuntimeError("num_classes could not be inferred from checkpoint; pass --num-classes")
                model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
                model.load_state_dict(sd, strict=False)
                model.to(device)
                return model
            except Exception:
                pass
        # other heuristics can be added
    raise RuntimeError(
        "Unable to automatically rebuild model from checkpoint. "
        "If you have a model builder in src.food101.model, ensure it exposes `build_model(num_classes=...)`. "
        "Alternatively, save a checkpoint that includes 'model' or a documented 'state_dict' format."
    )

# ---- plotting util ----
def plot_confusion_matrix(cm, classes, out_path: Path):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=6)
    plt.yticks(tick_marks, classes, fontsize=6)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_reliability(y_true, y_prob, out_path: Path, n_bins=10):
    # y_prob: probability assigned to predicted class (top-1 prob)
    # reliability diagram: calibration_curve gives mean_pred, fraction_of_positives
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    plt.figure(figsize=(6,6))
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0,1],[0,1], linestyle='--')
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability diagram")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

# ---- main eval routine ----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True, help="Root path that contains food-101/")
    p.add_argument("--out-dir", default="artifacts/eval")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-classes", type=int, default=None, help="Optional: force num classes")
    args = p.parse_args()

    checkpoint_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # transforms: ImageNet-style
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    print("Loading dataset...")
    ds = Food101TestDataset(args.data_root, split="test", transform=transform)
    classes = ds.idx_to_class
    if args.num_classes is None:
        NUM_CLASSES = len(classes)
    else:
        NUM_CLASSES = args.num_classes

    print(f"Found {len(ds)} test samples, {NUM_CLASSES} classes.")

    # try to load classes from checkpoint (for stable mapping)
    ck_classes = try_load_classes(checkpoint_path, out_dir)
    if ck_classes is not None:
        print("Found classes in checkpoint/sidecar; using those for label mapping.")
        # ck_classes may be dict or list
        if isinstance(ck_classes, dict):
            # assume class->idx
            try:
                idx_to_class = {int(v): k for k, v in ck_classes.items()}
            except Exception:
                # might be idx->class already
                idx_to_class = {int(k): v for k, v in ck_classes.items()}
        elif isinstance(ck_classes, list):
            idx_to_class = {i: c for i, c in enumerate(ck_classes)}
        else:
            idx_to_class = classes
    else:
        idx_to_class = classes

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("Building model...")
    model = None
    try:
        model = build_model_from_checkpoint(checkpoint_path, device, num_classes=NUM_CLASSES)
        # attempt to load checkpoint state dict if available separately
        ck = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ck, dict):
            if "state_dict" in ck or "model_state_dict" in ck:
                sd = ck.get("state_dict", ck.get("model_state_dict"))
                model.load_state_dict(sd, strict=False)
    except Exception as e:
        print("Model build failed:", e)
        print("Attempting to load checkpoint as full model object...")
        ck = torch.load(checkpoint_path, map_location=device)
        if isinstance(ck, torch.nn.Module):
            model = ck.to(device)
        elif isinstance(ck, dict) and "model" in ck and isinstance(ck["model"], torch.nn.Module):
            model = ck["model"].to(device)
        else:
            raise RuntimeError("Could not reconstruct model; please provide a checkpoint that contains either a serializable model or state_dict and ensure model builder exists in src.food101.model.") from e

    model.eval()
    model.to(device)

    y_true = []
    y_pred = []
    y_prob = []  # probability for predicted class (top-1 conf)
    records = []

    softmax = torch.nn.Softmax(dim=1)

    print("Running inference...")
    with torch.no_grad():
        for imgs, labels, paths in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = softmax(logits).cpu().numpy()
            preds = probs.argmax(axis=1)
            confs = probs.max(axis=1)
            for pth, gt, pr, cf, row_probs in zip(paths, labels.numpy(), preds, confs, probs):
                records.append({
                    "image": pth,
                    "true_idx": int(gt),
                    "pred_idx": int(pr),
                    "pred_conf": float(cf),
                    "pred_label": idx_to_class[int(pr)],
                    "true_label": idx_to_class[int(gt)],
                })
                y_true.append(int(gt))
                y_pred.append(int(pr))
                y_prob.append(float(cf))

    # metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # classification report
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # save artifacts
    import pandas as pd
    df_preds = pd.DataFrame(records)
    preds_csv = out_dir / "preds.csv"
    df_preds.to_csv(preds_csv, index=False)
    print(f"Wrote predictions to {preds_csv}")

    report_csv = out_dir / "classification_report.csv"
    pd.DataFrame(report).transpose().to_csv(report_csv)
    print(f"Wrote classification report to {report_csv}")

    cm_csv = out_dir / "confusion_matrix.csv"
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(cm_csv)
    print(f"Wrote confusion matrix CSV to {cm_csv}")

    cm_png = out_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, target_names, cm_png)
    print(f"Wrote confusion matrix image to {cm_png}")

    reliability_png = out_dir / "reliability.png"
    # For calibration, we need binary indicators. Use top-1 correctness as y_true_bin and top-1 prob
    y_true_bin = [int(a==b) for a,b in zip(y_true, y_pred)]
    if len(set(y_true_bin)) > 1:
        plot_reliability(y_true_bin, y_prob, reliability_png)
        print(f"Wrote reliability diagram to {reliability_png}")
    else:
        print("Skipping reliability diagram (no positive/negative variation).")

    summary = {
        "accuracy": accuracy,
        "n_samples": len(y_true),
        "out_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("Done. Artifacts saved to:", out_dir)

if __name__ == "__main__":
    main()
