import os

import torch
from PIL import Image

from data import read_splits, build_transforms, make_datasets
import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:'mode' parameter is deprecated and will be removed in Pillow 13.*:DeprecationWarning"
)

def _write_dummy_jpeg(path, size=(64, 64), color=(128, 64, 32)):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    img = Image.new("RGB", size, color)
    img.save(path, format="JPEG")


def _create_tiny_food101_layout(tmp_path):
    """
    Create a minimal Food-101-like layout under tmp_path:

        <tmp>/food-101/
            images/<class>/<file>.jpg
            meta/train.txt
            meta/test.txt

    Returns the data_root as a string.
    """
    tmp_root = str(tmp_path)
    data_root = os.path.join(tmp_root, "food-101")
    images_root = os.path.join(data_root, "images")
    meta_root = os.path.join(data_root, "meta")

    os.makedirs(meta_root, exist_ok=True)

    classes = ["class_a", "class_b"]
    train_lines = []
    test_lines = []

    # Two images per class: one train, one test
    for cls in classes:
        # train sample
        rel_train = f"{cls}/{cls}_train"
        _write_dummy_jpeg(os.path.join(images_root, rel_train + ".jpg"))
        train_lines.append(rel_train)

        # test sample
        rel_test = f"{cls}/{cls}_test"
        _write_dummy_jpeg(os.path.join(images_root, rel_test + ".jpg"))
        test_lines.append(rel_test)

    with open(os.path.join(meta_root, "train.txt"), "w", encoding="utf8") as f:
        f.write("\n".join(train_lines))
    with open(os.path.join(meta_root, "test.txt"), "w", encoding="utf8") as f:
        f.write("\n".join(test_lines))

    return data_root


def test_make_datasets_end_to_end(tmp_path):
    data_root = _create_tiny_food101_layout(tmp_path)

    meta_folder = os.path.join(data_root, "meta")
    image_folder = os.path.join(data_root, "images")

    train_list, test_list = read_splits(meta_folder)
    assert len(train_list) == 2
    assert len(test_list) == 2

    train_tf, val_tf = build_transforms(img_size=64)

    train_ds, val_ds, classes = make_datasets(
        image_folder,
        train_list,
        test_list,
        train_tf,
        val_tf,
    )

    # Dataset lengths and classes
    assert len(train_ds) == len(train_list)
    assert len(val_ds) == len(test_list)
    assert set(classes) == {"class_a", "class_b"}

    # A single sample goes through transforms correctly
    x0, y0 = train_ds[0]
    assert isinstance(x0, torch.Tensor)
    assert x0.shape[1:] == (64, 64)
    assert y0 in {0, 1}
