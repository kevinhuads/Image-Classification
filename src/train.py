# train.py
import os
import argparse
import random
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch import amp

from .data import read_splits, build_transforms, make_datasets
from .model import build_resnet50
from .engine import train_one_epoch, validate, save_checkpoint, append_csv

def set_seed(seed: int = 3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parse_args():
    p = argparse.ArgumentParser(description="Train head-only or full fine-tune on Food-101 (refactored)")
    p.add_argument("--data_folder", default=r"data", help="root dataset folder")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--freeze_backbone", action="store_true", help="freeze pretrained backbone (head-only)")
    p.add_argument("--pretrained", action="store_true", help="use ImageNet pretrained weights")
    p.add_argument("--output", default=None, help="path to save best checkpoint (defaults inside data_folder)")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(3)

    data_folder = args.data_folder
    image_folder = os.path.join(data_folder, "images")
    meta_folder = os.path.join(data_folder, "meta")
    os.makedirs(data_folder, exist_ok=True)

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

    output_path = args.output or os.path.join(data_folder, "refactored_best.pth")
    csv_path = os.path.join(data_folder, "refactored_results.csv")

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
