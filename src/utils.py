# utils.py
import io
from typing import List, Tuple
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# keep preprocessing identical across scripts
PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def load_model_and_meta(ckpt_path: str, device: torch.device | None = None, map_location:str="cuda"):
    """
    Load checkpoint (expects keys: 'model_state_dict', 'classes').
    Returns: model (eval), device, preprocess callable, classes list.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=map_location,weights_only=False)
    classes = ckpt["classes"]
    num_classes = len(classes)

    # build the architecture (match training script)
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    return model, device, PREPROCESS, classes

def predict_pil(image: Image.Image, model, device, preprocess, topk: int = 5) -> List[Tuple[str, float]]:
    """
    Run inference on a PIL image. Returns list of (class, prob).
    """
    x = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
        logits = model(x)
        
        # Ensure logits are in float32 for CPU compatibility
        if logits.dtype == torch.bfloat16:
            logits = logits.to(torch.float32)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    topk_idx = probs.argsort()[::-1][:topk]
    return [(int(idx), float(probs[idx])) for idx in topk_idx]  # returns indices + probs

def topk_labels(pred_idx_probs, classes):
    """Convert list of (idx, prob) to (label, prob)."""
    return [(classes[idx], prob) for idx, prob in pred_idx_probs]
