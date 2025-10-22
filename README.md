# Image Classification

A Python-based project for training, evaluating, and experimenting with image classification models. The repository includes reproducible environments, Jupyter notebooks for exploration, and a structured codebase for development and testing.

## Key Features

- Jupyter notebooks for rapid experimentation and visualization
- Reproducible environments via `requirements.txt` and a Dockerfile
- Organized project layout with `src/`, `configs/`, and `tests/`
- Lightweight app entry point (`app.py`) for demos or quick runs
- CI-friendly dependency pinning via `requirements-ci.txt`

## Project Structure

```
.
├─ .dockerignore           # Files/directories excluded from Docker context
├─ .gitignore              # Git ignore rules
├─ .github/                # GitHub configuration (e.g., workflows, templates)
├─ Dockerfile              # Containerized environment
├─ README.md               # You are here
├─ app.py                  # Minimal entry point (demo / script runner)
├─ configs/                # Configuration files (e.g., experiment settings)
├─ notebooks/              # Jupyter notebooks for analysis and prototyping
├─ pyproject.toml          # Project metadata and build config (PEP 621)
├─ requirements-ci.txt     # CI-focused dependency pinning
├─ requirements.txt        # Main project dependencies
├─ src/                    # Source code
└─ tests/                  # Test suite
```

## Getting Started

### 1) Prerequisites

- Python 3.11
- pip 

### 2) Clone the repository

```bash
git clone https://github.com/kevinhuads/Image-Classification.git
cd Image-Classification
```

### 3) Set up a virtual environment

Using `venv`:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 4) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```



## Usage

### Notebooks

- Explore and prototype in `notebooks/`.
- Recommend using the created kernel (`image-classification`) for consistent dependencies.

### Scripts and app

- The repository includes a minimal `app.py`. You can use it as a starting point for quick inference demos, utility scripts, or CLI entry points:

```bash
streamlit run app.py
```

- Explore the `src/` directory for reusable modules and components. Organize your training/evaluation workflows there.

### YAML-first commands for src modules

Prefer keeping all settings in YAML files under `configs/`, and invoke modules with only `--config` (CLI flags remain optional overrides if you need them).

- Training:
  ```bash
  python -m src.train --config configs/train.yaml
  ```

- Evaluation:
  ```bash
  python -m src.eval --config configs/eval.yaml
  ```

- Inference:
  ```bash
  python -m src.infer --config configs/infer.yaml
  ```

Minimum keys expected in the YAMLs:
- train.yaml: data_folder, output_folder, epochs, batch_size, lr, weight_decay, num_workers, seed, device, pretrained, freeze_backbone
- eval.yaml: checkpoint, data_root, out_dir, batch_size, num_workers, device, num_classes (optional)
- infer.yaml: image_path, ckpt, topk

### Configurations

- Store and version your experiment settings in `configs/`.
- Add new configuration files as needed (e.g., YAML/JSON for datasets, model hyperparameters, and training schedules).

## Docker

Build an image:

```bash
docker build -t image-classification:latest .
```

Run interactively and mount the project:

```bash
docker run --rm -it \
  -v "$PWD":/workspace \
  -w /workspace \
  image-classification:latest \
  bash
```

Optionally, run the app inside the container:

```bash
docker run --rm -it \
  -v "$PWD":/workspace \
  -w /workspace \
  image-classification:latest \
  python app.py
```

If your app serves a web UI/API, expose a port (replace 7860 with your app’s port):

```bash
docker run --rm -it -p 7860:7860 \
  -v "$PWD":/workspace \
  -w /workspace \
  image-classification:latest
```

## Testing

Run the test suite (assuming `pytest` is included in the dependencies):

```bash
pytest -v
```

Place new tests under `tests/` and follow a clear naming convention (e.g., `test_*.py`).

## Data Management

- Store datasets outside the repository or under a dedicated `data/` directory (ignored by Git if large).
- Consider using symlinks, environment variables, or configuration files in `configs/` to point to dataset locations.
- For large files, prefer external storage or Git LFS.

## Reproducibility Tips

- Pin dependencies where possible (see `requirements-ci.txt` for CI).
- Keep configurations in version control (`configs/`).
- Record random seeds in your training/evaluation scripts.
- Capture environment info (e.g., `pip freeze`, `python --version`) in experiment logs.