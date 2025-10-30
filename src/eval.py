#!/usr/bin/env python3
"""
Evaluation script for Food-101 models.

Usage example:
  python eval.py --config configs/eval.yaml
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.calibration import calibration_curve

from config import load_yaml

# ---- Dataset helper for Food-101 test split ----
class Food101TestDataset(Dataset):
    def __init__(self, data_root: str, split="test", transform=None):
        """
        Expects Food-101 layout:
        data_root/images/<class>/<image>.jpg
        data_root/meta/test.txt (each line like: 'apple_pie/12345')
        """
        self.data_root = Path(data_root)
        self.image_root = self.data_root / "images"
        meta_file = self.data_root / "meta" / f"{split}.txt"
        with open(meta_file, "r") as fh:
            lines = [l.strip() for l in fh.readlines() if l.strip()]
        self.samples = []
        self.idx_to_class = []
        class_set = {}
        for ln in lines:
            # ln like class/imageid
            cl, fn = ln.split("/")
            img_path = self.image_root / cl / f"{fn}.jpg"
            self.samples.append((str(img_path), cl))
            if cl not in class_set:
                class_set[cl] = len(class_set)
                self.idx_to_class.append(cl)
        self.class_to_idx = {c: i for i, c in enumerate(self.idx_to_class)}
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cl = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[cl]
        return img, label

def apply_yaml_to_args(args, yaml_cfg):
    """
    For every key in yaml_cfg, set attr on args only if CLI left it None.
    This gives CLI priority when a user explicitly overrides a YAML value.
    """
    for k, v in (yaml_cfg.items() if yaml_cfg else []):
        if not hasattr(args, k):
            # ignore unknown keys or add mapping if you renamed CLI args
            continue
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args

def build_model_from_checkpoint(checkpoint_path: Path, device: torch.device, num_classes=None):
    """
    Helper that reconstructs a ResNet-50 if needed and loads weights.
    (This is unchanged from prior behavior in this file.)
    """
    import torchvision.models as models
    sd = torch.load(checkpoint_path, map_location=device, weights_only = True)
    # Attempt to construct resnet50 if the checkpoint contains an fc weight
    if num_classes is None and "fc.weight" in sd:
        num_classes = sd["fc.weight"].shape[0]
    model = models.resnet50(weights=None)
    # adjust final layer if needed
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="path to YAML config")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--data-root", dest="data_root", default=None, help="Root path that contains food-101/")
    p.add_argument("--out-dir", dest="out_dir", default="artifacts/eval")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", dest="num_workers", type=int, default=4)
    p.add_argument("--num-classes", dest="num_classes", type=int, default=None, help="Optional: force num classes")
    args = p.parse_args()

    yaml_cfg = {}
    if args.config:
        yaml_cfg = load_yaml(args.config)

    args = apply_yaml_to_args(args, yaml_cfg)

    # now validate required fields (after YAML applied)
    if args.ckpt is None:
        raise ValueError("checkpoint must be provided via CLI or --config")
    if args.data_root is None:
        raise ValueError("data_root must be provided via CLI or --config")

    checkpoint_path = Path(args.ckpt)
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

    model = build_model_from_checkpoint(checkpoint_path, device, num_classes=NUM_CLASSES)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    y_true = []
    y_pred = []
    probs = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            p = torch.nn.functional.softmax(out, dim=1)
            top1 = p.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(top1)
            y_true.extend(yb.numpy().tolist())
            probs.extend(p.max(dim=1).values.cpu().numpy().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    print("Top-1 accuracy:", accuracy)

    # reliability / calibration plot (optional)
    try:
        prob_true, prob_pred = calibration_curve(np.array(y_true)==np.array(y_pred), np.array(probs), n_bins=10)
        fig, ax = plt.subplots()
        ax.plot(prob_pred, prob_true, marker="o")
        ax.plot([0,1],[0,1], linestyle="--")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Fraction correct")
        ax.set_title("Reliability diagram")
        fig.savefig(out_dir / "reliability.png")
    except Exception:
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
