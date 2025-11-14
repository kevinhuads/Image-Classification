import os
import argparse
import json
import random
import numpy as np
import torch
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch import amp

from config import load_yaml, apply_yaml_to_args, resolve_paths, set_seed, get_device_and_pin
from data import read_splits, build_transforms, make_datasets
from model import build_model
from utils import infer_one_batch_signature
from engine import train_one_epoch, validate, save_checkpoint, append_csv, evaluate_and_log
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
import contextlib
import subprocess, sys

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="path to YAML config")
    p.add_argument("--data_folder", type=str, default=None, help="root dataset folder")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--freeze_backbone", action="store_true", default=None)
    p.add_argument("--pretrained", action="store_true", default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--mlflow", action="store_true", help="Enable MLflow logging")
    p.add_argument("--mlflow_experiment", type=str, default="Image-Classification", help="MLflow experiment name")
    p.add_argument("--mlflow_tracking_uri", type=str, default=None, help="MLflow tracking URI (overrides env)")
    p.add_argument("--mlflow_tags",type=json.loads,default=None,help='MLflow tags as JSON dict, e.g. \'{"dataset":"food101","stage":"dev"}\'')
    p.add_argument("--arch", type=str, default=None, help="torchvision model name (e.g., resnet50, efficientnet_b0, convnext_tiny)")

    return p
    
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
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("artifacts/checkpoints", exist_ok=True)
    os.makedirs("artifacts/per_epoch", exist_ok=True)

    pt = ""
    if not args.pretrained:
        pt = "_noPT"
        
    fb = ""
    if not args.freeze_backbone:
        fb = "_noFB"
        
    bsize = ""
    if args.batch_size != 64:
        bsize = f"_b{args.batch_size}"
        
    run_name = f"{args.arch}{bsize}{pt}{fb}_{args.epochs}"
    ckpt_path = os.path.join("artifacts/checkpoints",f"{run_name}.pth")
    csv_path = os.path.join("artifacts/per_epoch",f"{run_name}.csv")

    train_list, test_list = read_splits(meta_folder)
    train_tf, val_tf = build_transforms(img_size=(224))
    train_ds, val_ds, classes = make_datasets(image_folder, train_list, test_list, train_tf, val_tf)

    device, device_str, pin_memory = get_device_and_pin()
    
    common_loader_kwargs = dict(num_workers=args.num_workers, pin_memory=False, persistent_workers=(args.num_workers or 0) > 0,prefetch_factor=2)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **common_loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,  **common_loader_kwargs)

    model = build_model(args.arch, num_classes=len(classes), pretrained=args.pretrained, freeze_backbone=args.freeze_backbone)
    model = model.to(device)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)
    scaler = amp.GradScaler(device=device_str)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                           steps_per_epoch=len(train_loader), epochs=args.epochs)
    
    
    run_ctx = contextlib.nullcontext()  # so code is clean if --mlflow is off
    if args.mlflow:
        if args.mlflow_tracking_uri:
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        run_ctx = mlflow.start_run(run_name=run_name)

    with run_ctx:
        if args.mlflow:
            # --- params & “static” metadata
            mlflow.log_params({
                "model": args.arch, 
                "pretrained": bool(args.pretrained),
                "freeze_backbone": bool(args.freeze_backbone),
                "batch_size": int(args.batch_size),
                "epochs": int(args.epochs),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "num_workers": int(args.num_workers),
                "seed": int(args.seed),
                "device": device.type,
                "n_classes": len(classes),
                "train_len": len(train_ds),
                "val_len": len(val_ds),
            })
            
            if hasattr(args, "mlflow_tags") and isinstance(args.mlflow_tags, dict):
                mlflow.set_tags(args.mlflow_tags)
    
        x_ex_np, signature = infer_one_batch_signature(model, val_loader, device)

        best_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            # train returns avg_loss and list of top1..topK accuracies
            train_loss, train_accs, train_lr, train_grad_norm, train_param_norm, train_epoch_time, train_ips = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, device, device_str, scheduler, topk_max=20
            )

            if args.mlflow:
                val_loss, val_accs, val_epoch_time, val_ips, y_true, y_score = validate(
                    model, val_loader, criterion, device, device_str, topk_max=20, collect=True
                )
                evaluate_and_log(model, val_loader, device, epoch, classes, run_name, mlflow_enabled=True,
                                 y_true=y_true, y_score=y_score)
            else:
                val_loss, val_accs, val_epoch_time, val_ips = validate(
                    model, val_loader, criterion, device, device_str, topk_max=20
                )


            # readable short print for console (show top-1 and top-5 for quick glance)
            print(
                f"Epoch {epoch} | Train loss {train_loss:.4f} acc1 {train_accs[0]:.4f} acc5 {train_accs[4]:.4f} "
                f"| Val loss {val_loss:.4f} acc1 {val_accs[0]:.4f} acc5 {val_accs[4]:.4f}"
            )

            # CSV row:
            # epoch, train_loss, train_acc1..train_acc20,
            # val_loss, val_acc1..val_acc20,
            # train_lr, train_grad_norm, train_param_norm,
            # train_time_sec, train_images_per_sec,
            # val_time_sec, val_images_per_sec
            csv_row = (
                [epoch, float(train_loss)]
                + [float(a) for a in train_accs]
                + [float(val_loss)]
                + [float(a) for a in val_accs]
                + [
                    float(train_lr),
                    float(train_grad_norm),
                    float(train_param_norm),
                    float(train_epoch_time),
                    float(train_ips),
                    float(val_epoch_time),
                    float(val_ips),
                ]
            )
            append_csv(csv_path, csv_row, topk_max=20)


            if args.mlflow:
                # prepare metrics dict for MLflow: include train/acc1..train/acc20 and val/acc1..val/acc20
                metrics_to_log = {
                    "train/loss": float(train_loss),
                    "val/loss": float(val_loss),
                    "train/lr": float(train_lr),
                    "train/grad_norm": float(train_grad_norm),
                    "train/param_norm": float(train_param_norm),
                    "train/epoch_time_sec": float(train_epoch_time),
                    "train/images_per_sec": float(train_ips),
                    "val/epoch_time_sec": float(val_epoch_time),
                    "val/images_per_sec": float(val_ips),
                }

                # add top-k train and val metrics
                for k in range(1, 21):
                    metrics_to_log[f"train/acc{k}"] = float(train_accs[k-1])
                    metrics_to_log[f"val/acc{k}"] = float(val_accs[k-1])

                mlflow.log_metrics(metrics_to_log, step=epoch)

            if val_accs[0] > best_acc:
                best_acc = val_accs[0]
                
                save_checkpoint(ckpt_path , epoch, model, optimizer, classes)  
                print("Saved best model", ckpt_path)

                if args.mlflow:
                    mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
                    mlflow.log_artifact(csv_path, artifact_path="metrics")

                    reqs = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
                    pip_reqs = [l for l in reqs.splitlines() if l.startswith(("torch==","torchvision==","numpy==","mlflow=="))]
                    mlflow.pytorch.log_model(model, name="model", input_example=x_ex_np, signature=signature,  pip_requirements=pip_reqs)

if __name__ == "__main__":
    main()
