import os
import sys
import textwrap
import subprocess
import yaml
from PIL import Image


def _write_dummy_jpeg(path, size=(32, 32), color=(128, 64, 32)):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    img = Image.new("RGB", size, color)
    img.save(path, format="JPEG")


def _create_tiny_food101_layout(tmp_path):
    """
    Create a minimal Food-101-like layout under tmp_path.

    Returns data_root as a string.
    """
    tmp_root = str(tmp_path)
    data_root = os.path.join(tmp_root, "food-101")
    images_root = os.path.join(data_root, "images")
    meta_root = os.path.join(data_root, "meta")

    os.makedirs(meta_root, exist_ok=True)

    classes = ["class_a", "class_b"]
    train_lines = []
    test_lines = []

    for cls in classes:
        # one train and one test sample per class
        rel_train = f"{cls}/{cls}_train"
        rel_test = f"{cls}/{cls}_test"

        _write_dummy_jpeg(os.path.join(images_root, rel_train + ".jpg"))
        _write_dummy_jpeg(os.path.join(images_root, rel_test + ".jpg"))

        train_lines.append(rel_train)
        test_lines.append(rel_test)

    with open(os.path.join(meta_root, "train.txt"), "w", encoding="utf8") as f:
        f.write("\n".join(train_lines))
    with open(os.path.join(meta_root, "test.txt"), "w", encoding="utf8") as f:
        f.write("\n".join(test_lines))

    return data_root


def test_train_cli_runs_one_epoch_on_tiny_data(tmp_path):
    """
    Run `python -m src.train --config <cfg>` on a minimal dataset and
    verify it exits successfully.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_root = _create_tiny_food101_layout(tmp_path)

    cfg_path = os.path.join(str(tmp_path), "train_tiny.yaml")
    cfg = {
        "arch": "tiny_cnn",
        "pretrained" : True,
        "data_folder": data_root,
        "batch_size": 2,
        "epochs": 1,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "num_workers": 4,       

        "freeze_backbone": True,
        "device": "cpu",
        "seed": 3,
        "mlflow": False,
    }

    with open(cfg_path, "w", encoding="utf8") as f:
        yaml.safe_dump(cfg, f)

    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [sys.executable, "-m", "src.train", "--config", cfg_path],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert proc.returncode == 0, (
        "src.train failed on tiny debug config.\n"
        f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
    )

    # Check that at least one checkpoint or metrics directory was produced.
    artifacts_dir = os.path.join(repo_root, "artifacts")
    assert os.path.exists(artifacts_dir), "Training did not create an artifacts/ directory."
