# Image-Classification

A compact, reproducible image classification project using the Food-101 dataset and modern PyTorch best practices. This repository provides a training and inference pipeline, developer tools, and a lightweight demo application.

---

## Contents

- **Overview** - what this repository implements and its intended use.
- **Quickstart** - a minimal set of commands to reproduce the baseline locally.
- **Installation** - environment and dependency management.
- **Data** - how to obtain and prepare the Food-101 dataset used in the repo.
- **Usage** - training, evaluation, inference, and demo commands.
- **Configuration** - where runtime settings live and how to override them.
- **Project layout** - explanation of important files and directories.
- **Reproducibility** - how experiments are tracked and artifacts stored.

---

## Overview

This project implements an image classification pipeline for the Food-101 dataset. The codebase is structured to separate library logic (model, data, training loop) from CLI entry points and demos. It is designed for research and development: to run experiments, compare backbones, and produce reproducible artifacts.

Intended audience: researchers and engineers who want a compact baseline to iterate on image-classification experiments.


## Quickstart (minimum steps)

> The repository assumes a Unix-like environment and a Python 3.10+ interpreter with access to pip or a project manager (Poetry).

1. Clone the repository:

```bash
git clone https://github.com/kevinhuads/Image-Classification.git
cd Image-Classification
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# or, if the project uses pyproject.toml with an installer, use:
# pip install -e .
```

3. Download and prepare the Food-101 dataset (see **Data** section).

4. Run a smoke training run (config shown as example):

```bash
python train.py --config config.yaml
```

5. Run inference on a single image:

```bash
python infer.py --checkpoint artifacts/checkpoints/best.pth --image tests/data/sample.jpg --topk 5
```

The Quickstart commands above are intentionally minimal. For more control, consult the **Usage** and **Configuration** sections below.


## Installation

Recommended options:

- **pip** (requirements file): `pip install -r requirements.txt`
- **Editable install** for development: `pip install -e .`
- **Poetry** (if a `pyproject.toml` and `poetry.lock` exist): `poetry install`.

Use a GPU-enabled environment if you plan to train the baseline model in reasonable time. The repository supports CPU-only execution for evaluation and small smoke tests.


## Data

### Dataset

This repository is configured for the Food-101 dataset. The dataset itself is not included in the repository due to licensing and size constraints.

A helper script `scripts/download_food101.py` (or equivalent) is provided to fetch and verify the dataset. The expected layout after preparation is:

```
data/
  food-101/
    images/
      apple_pie/
        00000001.jpg
        ...
    meta/
      train.txt
      test.txt
```

Place the dataset root path in the configuration (`data.root`) or supply it via CLI `--data-root`.

### Preprocessing

Training and evaluation use transforms that match the pretrained backbone normalization (ImageNet mean/std) and recommended resizing/cropping rules. See `src/food101/data.py` for explicit transform definitions.


## Usage

### Training

The primary training entry point is `train.py`. Basic usage:

```bash
python train.py --config config.yaml
```

Key CLI options:

- `--config` - path to YAML configuration file.
- `--checkpoint-dir` - directory where checkpoints and artifacts are saved.
- `--device` - `cuda` / `cpu` / `mps` (if supported).
- `--resume` - path to a checkpoint to resume training.

Training implements mixed-precision, checkpointing (best + last), seed logging, and experiment metadata preservation.


### Evaluation

A standalone `eval.py` computes metrics on a held-out test set and exports artifacts:

```bash
python eval.py --checkpoint artifacts/checkpoints/best.pth --data-root /path/to/data --out-dir artifacts/eval
```

The evaluation script emits per-class precision/recall/F1, a confusion matrix (CSV + image), and calibration diagnostics.


### Inference

`infer.py` performs prediction on single images or folders and writes results to JSON or CSV.

```bash
python infer.py --checkpoint artifacts/checkpoints/best.pth --image tests/data/sample.jpg --topk 5 --output preds.json
```

Options include `--topk`, `--confidence-threshold`, and `--batch-size` for folder inference.


### Demo (Streamlit)

A lightweight demo UI is provided at `app.py`. To run locally:

```bash
streamlit run app.py
```

The demo supports image upload, URL ingestion, and visual explanations (Grad-CAM) for the selected checkpoint.


## Configuration

Runtime configuration is centralized in `config.yaml`. The project supports overriding configuration values via CLI arguments (for example, `--data.root` and `--trainer.max_epochs`), or by using an alternative config file.

A representative `config.yaml` contains keys for:

- `data` - root path, batch size, workers, transforms.
- `model` - backbone name, pretrained flag, number of classes.
- `trainer` - optimizer, learning-rate schedule, epochs, mixed-precision.
- `logging` - experiment name, output directories, tracking backend.


## Project layout

```
.
├─ README.md                # this file
├─ config.yaml              # default runtime configuration
├─ requirements.txt         # pinned runtime deps for reproducibility
├─ pyproject.toml           # optional; packaging metadata
├─ train.py                 # CLI wrapper for training
├─ infer.py                 # CLI wrapper for inference
├─ eval.py                  # evaluation and metrics export
├─ app.py                   # Streamlit demo
├─ scripts/                 # utility scripts (data download, conversion)
├─ src/food101/             # library code: data, model, engine, utils
└─ tests/                   # unit and smoke tests
```


## Reproducibility and artifacts

Training runs produce reproducible artifacts saved under the configured `artifacts/` directory. Each run preserves:

- `checkpoints/` - `best.pth`, `last.pth` (and optional epoched checkpoints).
- `classes.json` - ordered label list used by the checkpoint.
- `config_used.yaml` - the fully resolved configuration for the run.
- `env.txt` - environment snapshot (`pip freeze`) captured after training.
- `metrics/` - CSV and plots for loss/accuracy, confusion matrix, and other diagnostics.

Each checkpoint includes metadata (git commit hash, config snapshot, class labels) to ensure correct usage during inference.


## Testing and quality

The repository includes a small test suite (`tests/`) with unit and smoke tests. Run the full test suite with:

```bash
pytest -q
```

Pre-commit hooks for code formatting and linting are recommended (see `.pre-commit-config.yaml`).