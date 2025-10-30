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
from model import build_resnet50
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
    p.add_argument("--model_path", type=str, default=None, help="output path for the model")
    p.add_argument("--csv_path", type=str, default=None, help="output path for the csv")
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
    p.add_argument("--mlflow_run_name", type=str, default=None, help="MLflow run_name")
    p.add_argument("--mlflow_tags",type=json.loads,default=None,help='MLflow tags as JSON dict, e.g. \'{"dataset":"food101","stage":"dev"}\'')
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

    train_list, test_list = read_splits(meta_folder)
    train_tf, val_tf = build_transforms()
    train_ds, val_ds, classes = make_datasets(image_folder, train_list, test_list, train_tf, val_tf)

    device, device_str, pin_memory = get_device_and_pin()
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=pin_memory)

    model = build_resnet50(num_classes=len(classes), pretrained=args.pretrained, freeze_backbone=args.freeze_backbone)
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
        run_ctx = mlflow.start_run(run_name=args.mlflow_run_name)

    with run_ctx:
        if args.mlflow:
            # --- params & “static” metadata
            mlflow.log_params({
                "model": "resnet50",
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
            train_loss, train_acc1, train_acc5, train_lr, train_grad_norm, train_param_norm, train_epoch_time, train_ips = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, device, device_str, scheduler
            )

            val_loss, val_acc1, val_acc5 = validate(model, val_loader, criterion, device, device_str)

            print(f"Epoch {epoch} | Train loss {train_loss:.4f} acc1 {train_acc1:.4f} acc5 {train_acc5:.4f} "
                f"| Val loss {val_loss:.4f} acc1 {val_acc1:.4f} acc5 {val_acc5:.4f}")

            append_csv(args.csv_path, [epoch, train_loss, train_acc1, train_acc5, val_loss, val_acc1, val_acc5])
            
            if args.mlflow:
                mlflow.log_metrics({
                    "train/loss": float(train_loss),
                    "train/acc1": float(train_acc1),
                    "train/acc5": float(train_acc5),
                    "val/loss": float(val_loss),
                    "val/acc1": float(val_acc1),
                    "val/acc5": float(val_acc5),
                    "train/lr": float(train_lr),
                    "train/grad_norm": float(train_grad_norm),
                    "train/param_norm": float(train_param_norm),
                    "train/epoch_time_sec": float(train_epoch_time),
                    "train/images_per_sec": float(train_ips),
                }, step=epoch)
                
                eval_res = evaluate_and_log(model, val_loader, device, epoch, classes, mlflow_enabled=True)

            if val_acc1 > best_acc:
                best_acc = val_acc1
                save_checkpoint(args.model_path, epoch, model, optimizer, classes)  # from engine.py
                print("Saved best model", args.model_path)

                if args.mlflow:
                    mlflow.log_artifact(args.model_path, artifact_path="checkpoints")
                    if args.csv_path is not None and os.path.exists(args.csv_path):
                        mlflow.log_artifact(args.csv_path, artifact_path="metrics")
                    try:
                        reqs = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
                        pip_reqs = [l for l in reqs.splitlines() if l.startswith(("torch==","torchvision==","numpy==","mlflow=="))]
                        mlflow.pytorch.log_model(model, name="model", input_example=x_ex_np, signature=signature,  pip_requirements=pip_reqs)
                    except Exception as e:
                        print(f"Skipping mlflow.pytorch.log_model: {e}")

if __name__ == "__main__":
    main()
