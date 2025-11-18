## Multi Classes Image Classification on Food Dataset


A Python-based project for training, evaluating, and deploying image classification models on the **Food-101** dataset.  
The task is a **multi-class classification** problem with **101 food categories** and approximately **101 000 images**, where each image belongs to exactly one class.

The project combines:

- **Exploratory Data Analysis (EDA)** of the dataset and its visual structure in embedding space.
- A **benchmark of modern architectures** (CNNs and Vision Transformers) under both frozen-backbone and full fine-tuning regimes.
- An **MLOps-ready stack** for training, tracking and deployment.

### Live demo
A hosted version of the application is available on Hugging Face Spaces.  
It runs as a Docker Space built from the same container image defined in this repository: 

https://kevinhuads-deepvision-workflow.hf.space/

## High-level results and model choices

The main experimental results are presented in the notebooks:

- `notebooks/1_EDA.ipynb`  
  Large-scale exploratory analysis of Food-101: image dimensions, aspect ratios, pretrained feature embeddings, t-SNE projections and hierarchical clustering. This notebook establishes how the dataset is organised in a generic pretrained feature space.

- `notebooks/2_models.ipynb`  
  Benchmark of several architectures, including MLP and small CNN baselines, ResNet50, EfficientNet, ConvNeXt, Vision Transformers and Swin Transformers. Models are evaluated with both frozen backbones and full fine-tuning on Food-101.

- `notebooks/3_MLOps.ipynb`  
  Description of the operational workflow around the project: MLflow based experiment tracking, Streamlit demo architecture, Docker and Docker Compose setup, and CI/CD pipeline based on GitHub Actions.

Under the current configuration:

- **swin_b** is the best performing model, reaching about **91.8% Top-1 accuracy** and **over 99% Top-7 accuracy** on the validation set. Its embeddings show strong class separation and high clustering quality after fine-tuning.
- **swin_t** achieves very similar performance (around **91.2% Top-1 accuracy**) with a significantly smaller checkpoint and faster inference and is used as the backbone for the **Streamlit application** (`app.py`), where its lighter footprint and faster predictions provide a more responsive user experience while preserving strong accuracy.

## Dataset

The project uses the **Food-101** dataset:

- 101 food categories (such as *pizza*, *sushi*, *steak*, *ramen*).
- 1 000 images per class, split into 750 training images and 250 test images.
- Images stored in a class-based folder structure with `meta/train.txt` and `meta/test.txt` defining the splits.

The notebooks and training scripts assume this layout under a configurable `data_folder` (for example `data/food-101`).


## Key Features

- Jupyter notebooks for EDA, model benchmarking and interpretation:
  - `1_EDA.ipynb` for dataset and representation analysis.
  - `2_models.ipynb` for architecture comparison and detailed study of **swin_b**.
  - `3_MLOps.ipynb` for an overview of experiment tracking, the Streamlit demo, containerisation and CI/CD.
- Reproducible environments via `requirements.txt` and a Dockerfile.
- Organised project layout with `src/`, `configs/`, and `tests/`.
- Streamlit application (`app.py`) for interactive inference with a lightweight swin_t backbone.
- CI-friendly dependency pinning via `requirements-ci.txt`.
- Integration points for MLflow, Docker and CI/CD to support an end-to-end MLOps workflow.


## Project Structure

```text
.
├─ .dockerignore           # Files/directories excluded from Docker context
├─ .gitignore              # Git ignore rules
├─ .github/                # GitHub configuration (workflows, templates)
├─ Dockerfile              # Containerised environment
├─ README.md               # You are here
├─ app.py                  # Streamlit entry point for interactive inference
├─ configs/                # Configuration files (experiment settings)
├─ notebooks/              # Jupyter notebooks (EDA, models, MLOps)
├─ pyproject.toml          # Project metadata and build config (PEP 621)
├─ requirements-ci.txt     # CI-focused dependency pinning
├─ requirements.txt        # Main project dependencies
├─ src/                    # Source code (training, evaluation, inference, app)
└─ tests/                  # Test suite
```

### Source code overview (`src/`)

- `app.py` - Streamlit web application for interactive Food-101 predictions and project overview.
- `config.py` - Utilities for loading YAML configuration, merging with CLI arguments, resolving paths, and selecting device / seeds.
- `data.py` - Helpers to read Food-101 splits, build preprocessing transforms, and construct train/validation/test datasets.
- `engine.py` - Core training and validation loops, metric computation, calibration analysis, and artifact logging.
- `eval.py` - Standalone script to evaluate a trained checkpoint on the Food-101 test split and generate summary plots/files.
- `infer.py` - Command line entry point for single image inference, combining YAML configuration, model loading and top-k prediction printing.
- `model.py` - Model factory that instantiates CNN and transformer backbones and replaces their classifier heads for Food-101.
- `styles.py` - CSS definitions used to theme the Streamlit demo interface.
- `train.py` - Command line training entry point that wires together config, data pipeline, model creation, engine, and MLflow logging.
- `utils.py` - Inference utilities for loading models and metadata, applying preprocessing, and computing top-k predictions from images.


## Getting Started

### 1) Prerequisites

- Python 3.11
- pip 

### 2) Clone the repository

```bash
git clone https://github.com/kevinhuads/deepvision-workflow.git
cd deepvision-workflow
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

The main analysis and experiments are documented in `notebooks/`:

- `1_EDA.ipynb`: dataset exploration, embeddings, t-SNE, dendrograms.
- `2_models.ipynb`: model benchmark (MLP, CNNs, transformers), comparison of frozen vs full fine-tuning, focus on swin_b.
- `3_MLOps.ipynb`: operational view of the project, covering MLflow tracking, the Streamlit application, Docker and Docker Compose configuration, and the GitHub Actions based CI/CD pipeline.

Running these notebooks with the same environment ensures reproducibility of the reported results.

### Scripts and app

The repository includes a Streamlit application for interactive inference where the user can test the model by uploading their own images. The model will then return the result with its confidence:

<img src="figures/screenshot_streamlit.png" width="800" alt="screenshot_streamlit">

To run it, run the following command:

```bash
streamlit run app.py
```

The app loads a trained swin_t checkpoint and provides:

- Image upload.
- Top-N predictions with probabilities.
- Visualisation of model confidence.

The core training, evaluation and inference logic is implemented in `src/`. The main entry points are:

### YAML-first commands for src modules

Configuration is handled via YAML files under `configs/`. Command-line arguments can be used as optional overrides.

- Training:
  ```bash
  python -m src.train --config configs/train.yaml
  ```

- Evaluation:
  ```bash
  python -m src.eval --config configs/eval.yaml
  ```

- Inference (CLI):
  ```bash
  python -m src.infer --config configs/infer.yaml
  ```

Minimum keys expected in the YAMLs:

- `train.yaml`: `data_folder`, `output_folder`, `epochs`, `batch_size`, `lr`, `weight_decay`, `num_workers`, `seed`, `device`, `pretrained`, `freeze_backbone`.
- `eval.yaml`: `ckpt`, `data_root`, `out_dir`, `batch_size`, `num_workers`, `device`, `num_classes` (optional).
- `infer.yaml`: `image_path`, `ckpt`, `topk`.

### Configurations

- Store and version experiment settings in `configs/`.
- Add new configuration files as needed for alternative datasets, model variants or training schedules.

## Docker

### Build and run

You can either build the image locally or use the prebuilt image published on Docker Hub.

### Option 1: Build and run locally

```bash
docker compose up --build
```

Run in background:

```bash
docker compose up -d
```

### Option 2: Use the prebuilt image

Pull the latest image:

```bash
docker pull kevinhuads/deepvision-workflow:latest
```

Run it directly:

```bash
docker run --rm -p 8501:8501  "kevinhuads/deepvision-workflow:latest" 
```

### Access

Open [http://localhost:8501](http://localhost:8501) to access the Streamlit app.

## Testing

Run the test suite:

```bash
pytest -v
```

The tests in `tests/` cover:
- Smoke checks for project layout and CLIs (imports, --help, basic parsing).
- Data pipeline contracts and runtime on a tiny synthetic Food-101–style dataset.
- Inference utilities (preprocessing, prediction API, top-k outputs).
- Metric helpers for accuracy and probability handling.
- A short “happy path” training run that produces artifacts on a minimal dataset.

## Data Management

- Store datasets under a dedicated `data/` directory (ignored by Git if large) or in an external location.
- Use configuration files in `configs/` to point to dataset locations.
- For large files or models, consider external storage or Git LFS.

## Reproducibility

- Pin dependencies where appropriate (see `requirements-ci.txt` for CI).
- Keep configurations under version control.
- Record random seeds in training and evaluation scripts.
- Log environment information (for example `pip freeze`, `python --version`) in experiment tracking tools such as MLflow.

## Limitations and business-oriented considerations

### Technical and data-related limitations

- **Dataset coverage**: Food-101 is a curated academic dataset with 101 predefined classes. Many real-world dishes, regional variants or mixed plates are not represented. The model is therefore not a universal food recogniser and may misclassify items outside this closed set of classes.
- **Single-label classification**: Each image is assumed to contain a single dominant dish. Multi-label situations (for example a plate combining salad, meat and sides, or buffet-style images) are not modelled explicitly and can lead to unstable or ambiguous predictions.
- **Domain shift and robustness**: The model is trained on Food-101 images, which differ from many production environments (lighting, camera quality, plating, user-generated cropping, etc.). Performance may degrade significantly on domains that differ from the training distribution without explicit adaptation.
- **No automatic nutritional or health information**: The system recognises dish categories only. It does not estimate portion size, ingredients, allergens or nutritional values and must not be used as a source of medical, dietary or allergen advice.
- **Demo-oriented serving stack**: The Streamlit application and Docker setup are designed for experimentation and demonstration. They do not include production-grade concerns such as autoscaling, high-availability, request authentication or detailed runtime monitoring.

### Business-oriented context

- **Example use cases**: The current system can support use cases such as semi-automated photo labelling for food platforms, menu search by image, or internal prototyping for food logging applications. For any of these, additional product work is needed around UX, error handling and human-in-the-loop review.
- **Operational metrics beyond accuracy**: In a real deployment, key metrics would include latency (for example p95 response time per image), throughput, system uptime and user-facing error rates, in addition to classification accuracy and calibration. These metrics are only partially explored in this project.
- **Monitoring and feedback loop**: A production system would log requests and predictions, sample and review failures, monitor class distribution drift and trigger model re-training when performance deteriorates. The current project describes these aspects conceptually but does not implement a full monitoring pipeline.
- **Governance and compliance**: Any integration into a customer-facing product would need additional work on data governance, privacy, consent and model documentation (for example extended model cards and risk assessments) aligned with the organisation’s policies and applicable regulations.
