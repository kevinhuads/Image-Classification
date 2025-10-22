import os
import argparse
import random
import numpy as np
import torch
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch import amp

from config_loader import load_yaml, merge_yaml_with_cli
from data import read_splits, build_transforms, make_datasets
from model import build_resnet50
from engine import train_one_epoch, validate, save_checkpoint, append_csv


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="path to YAML config")
    p.add_argument("--data_folder", type=str, default=None, help="root dataset folder")
    p.add_argument("--output_folder", type=str, default=None, help="output folder")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--freeze_backbone", action="store_true", default=None)
    p.add_argument("--pretrained", action="store_true", default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p

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

def resolve_paths(args):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if args.data_folder is None:
        raise ValueError("data_folder must be provided (via CLI or --config).")
    data_folder = Path(args.data_folder).expanduser()
    if not data_folder.is_absolute():
        data_folder = (PROJECT_ROOT / data_folder).resolve()
    args.data_folder = str(data_folder)

    # coerce numeric fields that may come from YAML as strings
    if args.seed is not None:
        args.seed = int(args.seed)
    if args.num_workers is not None:
        args.num_workers = int(args.num_workers)
    if args.batch_size is not None:
        args.batch_size = int(args.batch_size)
    if args.epochs is not None:
        args.epochs = int(args.epochs)
    if args.lr is not None:
        args.lr = float(args.lr)
    if args.weight_decay is not None:
        args.weight_decay = float(args.weight_decay)
    return args


def set_seed(seed: int = 3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def main():
    
    
    parser = build_parser()
    args = parser.parse_args()
    
    yaml_cfg = {}
    if args.config:
        yaml_cfg = load_yaml(args.config)

    args = apply_yaml_to_args(args, yaml_cfg)
    args = resolve_paths(args)
    set_seed(args.seed)

    image_folder = os.path.join(args.data_folder, "images")
    meta_folder = os.path.join(args.data_folder, "meta")
    os.makedirs(args.output_folder, exist_ok=True)


    train_list, test_list = read_splits(meta_folder)
    train_tf, val_tf = build_transforms()
    train_ds, val_ds, classes = make_datasets(image_folder, train_list, test_list, train_tf, val_tf)

    has_cuda = torch.cuda.is_available()
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    use_pinned = has_cuda or has_xpu

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=use_pinned)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=use_pinned)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = device.type

    model = build_resnet50(num_classes=len(classes), pretrained=args.pretrained, freeze_backbone=args.freeze_backbone)
    model = model.to(device)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)
    scaler = amp.GradScaler(device=device_str)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                           steps_per_epoch=len(train_loader), epochs=args.epochs)

    output_path = os.path.join(args.output_folder, "refactored_best.pth")
    csv_path = os.path.join(args.output_folder, "refactored_results.csv")

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, device_str, scheduler
        )
        val_loss, val_acc1, val_acc5 = validate(model, val_loader, criterion, device, device_str)

        print(f"Epoch {epoch} | Train loss {train_loss:.4f} acc1 {train_acc1:.4f} acc5 {train_acc5:.4f} "
              f"| Val loss {val_loss:.4f} acc1 {val_acc1:.4f} acc5 {val_acc5:.4f}")

        append_csv(csv_path, [epoch, train_loss, train_acc1, train_acc5, val_loss, val_acc1, val_acc5])

        if val_acc1 > best_acc:
            best_acc = val_acc1
            save_checkpoint(output_path, epoch, model, optimizer, classes)
            print("Saved best model", output_path)

if __name__ == "__main__":
    main()
