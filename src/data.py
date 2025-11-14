# data.py
import os
from typing import List, Tuple
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_splits(meta_folder: str) -> Tuple[List[str], List[str]]:
    with open(os.path.join(meta_folder, "train.txt")) as f:
        train_list = [line.strip() + ".jpg" for line in f]
    with open(os.path.join(meta_folder, "test.txt")) as f:
        test_list = [line.strip() + ".jpg" for line in f]
    return train_list, test_list

def build_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    # keep ImageNet normalization for both modes (works fine from scratch too)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    # scale Resize proportionally to classic 256->224 pipeline
    val_resize = max(img_size + 32, int(round(img_size * 256 / 224)))
    val_tf = transforms.Compose([
        transforms.Resize(val_resize),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tf, val_tf

def make_datasets(image_folder: str, train_list: List[str], test_list: List[str],
                  train_transform: transforms.Compose, val_transform: transforms.Compose):
    # Build class list from the union of splits (same approach as before)
    classes = sorted({fp.split('/')[0] for fp in train_list + test_list})
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_samples = [(os.path.join(image_folder, fp), class_to_idx[fp.split('/')[0]]) for fp in train_list]
    val_samples   = [(os.path.join(image_folder, fp), class_to_idx[fp.split('/')[0]]) for fp in test_list]

    train_ds = DatasetFolder(root=image_folder, loader=default_loader, extensions=("jpg",), transform=train_transform)
    train_ds.samples = train_samples
    train_ds.targets = [s[1] for s in train_samples]
    train_ds.classes = classes
    train_ds.class_to_idx = class_to_idx

    val_ds = DatasetFolder(root=image_folder, loader=default_loader, extensions=("jpg",), transform=val_transform)
    val_ds.samples = val_samples
    val_ds.targets = [s[1] for s in val_samples]
    val_ds.classes = classes
    val_ds.class_to_idx = class_to_idx

    return train_ds, val_ds, classes
