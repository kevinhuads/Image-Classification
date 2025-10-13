# model.py
import torch.nn as nn
from torchvision import models

def build_resnet50(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    """
    Returns a ResNet-50 with final fc replaced for num_classes.
    If pretrained=True, uses models.ResNet50_Weights.DEFAULT.
    If freeze_backbone=True, sets requires_grad=False for backbone parameters.
    """
    if pretrained:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)

    # Freeze backbone optionally (leave fc trainable)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
