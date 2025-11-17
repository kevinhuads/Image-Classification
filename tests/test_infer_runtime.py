from PIL import Image
import torch
import torch.nn as nn

from utils import PREPROCESS, predict_pil, topk_labels


class DummyNet(nn.Module):
    """
    Very small model that produces logits of shape (B, num_classes)
    so that predict_pil + softmax work as expected.
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x):
        # x: (B, 3, H, W) -> (B, num_classes)
        x = self.conv(x)
        return x.mean(dim=(2, 3))


def _dummy_image():
    return Image.new("RGB", (224, 224), color=(200, 100, 50))


def test_predict_pil_returns_topk_indices_and_probs():
    device = torch.device("cpu")
    model = DummyNet(num_classes=3).to(device)
    model.eval()

    img = _dummy_image()

    topk = 2
    idx_probs = predict_pil(img, model, device, PREPROCESS, topk=topk)

    # Structure: list[(int, float)] of length topk
    assert isinstance(idx_probs, list)
    assert len(idx_probs) == topk
    for idx, prob in idx_probs:
        assert isinstance(idx, int)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    # Probabilities must be sorted in descending order
    probs_only = [p for _, p in idx_probs]
    assert probs_only == sorted(probs_only, reverse=True)


def test_topk_labels_maps_indices_to_labels():
    # pretend we got these from predict_pil
    idx_probs = [(0, 0.7), (2, 0.2)]
    classes = ["class_a", "class_b", "class_c"]

    labeled = topk_labels(idx_probs, classes)

    assert labeled == [("class_a", 0.7), ("class_c", 0.2)]
