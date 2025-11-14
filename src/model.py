# model.py
import torch
import torch.nn as nn
from torchvision import models

class TinyCNN(nn.Module):
    """~200–500k params depending on widths; good as a scratch baseline."""
    def __init__(self, num_classes: int, in_ch: int = 3, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2

            nn.Conv2d(width, 2*width, 3, padding=1), nn.BatchNorm2d(2*width), nn.ReLU(inplace=True),
            nn.Conv2d(2*width, 2*width, 3, padding=1), nn.BatchNorm2d(2*width), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4

            nn.Conv2d(2*width, 4*width, 3, padding=1), nn.BatchNorm2d(4*width), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(4*width, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class SimpleMLP(nn.Module):
    """
    MLP that infers its input dimension dynamically on first forward() call
    or via lazy_build().
    """
    def __init__(self, num_classes: int, hidden: int = 512, p: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.hidden = hidden
        self.p = p
        self.flatten = nn.Flatten()
        self.net = None  # built lazily

    def _build(self, in_dim):
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.p),
            nn.Linear(self.hidden, self.num_classes)
        )

    def lazy_build(self, example_shape):
        """Initialize layers once if input shape is known."""
        if self.net is None:
            c, h, w = example_shape
            in_dim = c * h * w
            self._build(in_dim)

    def forward(self, x):
        x = self.flatten(x)
        if self.net is None:
            self._build(x.shape[1])
        return self.net(x)

    
def _replace_classifier(model, arch: str, num_classes: int):
    a = arch.lower()

    # --- ResNet / ResNeXt / WideResNet / RegNet / GoogLeNet / Inception: single fc named 'fc' ---
    if hasattr(model, "fc") and isinstance(getattr(model, "fc"), nn.Linear):
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)

        # If there's an auxiliary classifier (Inception/GoogLeNet), replace its fc too (if present)
        # TorchVision names it AuxLogits (Inception) or aux1/aux2 (older wrappers). Guard defensively.
        if hasattr(model, "AuxLogits") and getattr(model, "AuxLogits") is not None:
            if hasattr(model.AuxLogits, "fc") and isinstance(model.AuxLogits.fc, nn.Linear):
                model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        if hasattr(model, "aux1") and getattr(model, "aux1") is not None:
            if hasattr(model.aux1, "fc") and isinstance(model.aux1.fc, nn.Linear):
                model.aux1.fc = nn.Linear(model.aux1.fc.in_features, num_classes)
        if hasattr(model, "aux2") and getattr(model, "aux2") is not None:
            if hasattr(model.aux2, "fc") and isinstance(model.aux2.fc, nn.Linear):
                model.aux2.fc = nn.Linear(model.aux2.fc.in_features, num_classes)
        return

    # --- EfficientNet / MobileNet / MNASNet / ConvNeXt: classifier modules ---
    if a.startswith("efficientnet") or a.startswith("mnasnet") or a.startswith("mobilenet_v2"):
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return
    if a.startswith("mobilenet_v3"):
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return
    if a.startswith("convnext"):
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return

    # --- DenseNet ---
    if a.startswith("densenet"):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return

    # --- VGG / AlexNet (classifier is nn.Sequential with last Linear) ---
    if a.startswith(("vgg", "alexnet")):
        last = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last.in_features, num_classes)
        return

    # --- SqueezeNet: conv classifier ---
    if a.startswith("squeezenet"):
        conv = model.classifier[1]  # Conv2d(in_ch, 1000, kernel_size=1)
        model.classifier[1] = nn.Conv2d(conv.in_channels, num_classes, kernel_size=1)
        return

    # --- Vision transformer / Swin head replacements ---
    if a.startswith("vit"):
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return
    if a.startswith("swin"):
        model.head = nn.Linear(model.head.in_features, num_classes)
        return

    raise ValueError(f"Unknown head pattern for arch='{arch}'")

class _ZeroAux(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = int(num_classes)
    def forward(self, x):
        # returns zeros of shape (B, num_classes) on same device/dtype as x
        return x.new_zeros((x.size(0), self.num_classes))

def _try_weight_enum(enum_name: str):
    # returns DEFAULT enum value if present, else None
    enum = getattr(models, enum_name, None)
    return getattr(enum, "DEFAULT", None) if enum is not None else None

def _default_weights_for(arch: str):
    # TorchVision 0.15+ exposes this helper; fall back otherwise.
    try:
        get_model_weights = getattr(models, "get_model_weights", None)
        if callable(get_model_weights):
            enum = get_model_weights(arch)
            return getattr(enum, "DEFAULT", None)
    except Exception:
        pass
    # Fallback: guarded static map (no DeiT here)
    CANDIDATES = {
        # Classic CNNs
        "resnet18": "ResNet18_Weights",
        "resnet50": "ResNet50_Weights",
        "resnet101": "ResNet101_Weights",
        "resnext50_32x4d": "ResNeXt50_32X4D_Weights",
        "wide_resnet50_2": "Wide_ResNet50_2_Weights",
        "densenet121": "DenseNet121_Weights",
        "vgg16": "VGG16_Weights",
        "alexnet": "AlexNet_Weights",
        # Efficient / mobile
        "efficientnet_b0": "EfficientNet_B0_Weights",
        "efficientnet_b3": "EfficientNet_B3_Weights",
        "efficientnet_b7": "EfficientNet_B7_Weights",
        "mobilenet_v2": "MobileNet_V2_Weights",
        "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
        "shufflenet_v2_x1_0": "ShuffleNet_V2_X1_0_Weights",
        "squeezenet1_1": "SqueezeNet1_1_Weights",
        "mnasnet1_0": "MNASNet1_0_Weights",
        # Modern CNNs
        "convnext_small": "ConvNeXt_Small_Weights",
        "regnet_y_400mf": "RegNet_Y_400MF_Weights",
        "regnet_x_8gf": "RegNet_X_8GF_Weights",
        # Transformers (in torchvision)
        "vit_b_16": "ViT_B_16_Weights",
        "swin_t": "Swin_T_Weights",
        "swin_b": "Swin_B_Weights",
        # Specialized
        "inception_v3": "Inception_V3_Weights",
        "googlenet": "GoogLeNet_Weights",
    }
    enum_name = CANDIDATES.get(arch)
    return _try_weight_enum(enum_name) if enum_name else None

def build_model(arch: str, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    
    if arch == "tiny_cnn":
        return TinyCNN(num_classes=num_classes)
    if arch == "mlp":
            model = SimpleMLP(num_classes=num_classes)
            # Pre-build using standard 224×224 RGB input to have parameters for optimizer
            model.lazy_build((3, 224, 224))
            return model
    
    arch = arch.lower()
    ctor = getattr(models, arch, None)
    if ctor is None:
        raise ValueError(f"Unknown architecture: {arch}")

    # pick weights enum if available
    weights = None
    if pretrained:
        try:
            enum = models.get_model_weights(arch)
            weights = getattr(enum, "DEFAULT", None)
        except Exception:
            weights = None

    # ---- handle aux-head architectures explicitly ----
    if arch in {"inception_v3", "googlenet"}:
        if weights is None:
            # construct from scratch without aux heads
            model = ctor(weights=None, aux_logits=False)
        else:
            # construct with weights (wrapper enforces aux_logits=True)
            model = ctor(weights=weights)

            # replace aux modules with a safe no-op that won't run convs on tiny feature maps
            # Inception uses model.AuxLogits; GoogLeNet uses aux1 / aux2
            try:
                # set the runtime flag to False (defensive)
                if hasattr(model, "aux_logits"):
                    model.aux_logits = False
            except Exception:
                pass

            # replace/neutralize the aux modules (so _forward won't try to run heavy convs)
            if hasattr(model, "AuxLogits"):
                model.AuxLogits = _ZeroAux(num_classes)
            if hasattr(model, "aux1"):
                model.aux1 = _ZeroAux(num_classes)
            if hasattr(model, "aux2"):
                model.aux2 = _ZeroAux(num_classes)
    else:
        model = ctor(weights=weights) if weights is not None else ctor(weights=None)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    _replace_classifier(model, arch, num_classes)
    return model
