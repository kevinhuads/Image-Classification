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

def accuracy_topk(output: torch.Tensor, target: torch.Tensor, topk=(1,5)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        res.append(correct[:k].reshape(-1).float().sum(0, keepdim=True).item())
    return res  # list [top1_count, top5_count]

def train_one_epoch(model: nn.Module, loader, optimizer: Optimizer, criterion: nn.Module,
                    scaler: amp.GradScaler, device: torch.device, device_str: str, scheduler: OneCycleLR = None):
    model.train()
    running_loss = 0.0
    running_top1 = 0
    running_top5 = 0
    total = 0

    for images, targets in tqdm(loader, desc="train"):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        with amp.autocast(device_type=device_str):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        # 1) update params
        scaler.step(optimizer)
        # 2) update LR right after optimizer.step()
        if scheduler is not None:
            scheduler.step()
        # 3) update scaler
        scaler.update()

        running_loss += loss.item() * images.size(0)
        t1, t5 = accuracy_topk(outputs, targets, topk=(1,5))
        running_top1 += t1
        running_top5 += t5
        total += images.size(0)

    return running_loss / total, running_top1 / total, running_top5 / total

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
