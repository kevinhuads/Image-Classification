# Test Suite Overview

This directory contains the automated tests for the image classification project.  
The tests are designed to validate:

- The public API and contracts of the core modules.
- The ability to run the training, evaluation and inference CLIs.
- The behaviour of data loading, preprocessing and metrics utilities.
- The presence of the Streamlit demo entrypoint and minimal runtime wiring.

All tests are written with `pytest`.

---

## How to run the tests

From the repository root (where the `src/` directory is located), run:

```bash
pytest -q tests
```

or, for more verbose output:

```bash
pytest -v tests
```

The tests assume:

- Python is started in the repository root.
- The `src/` directory contains the project modules.
- The dependencies listed in the project requirements are installed.

---

## File-by-file description

### `conftest.py`

Shared configuration for the test suite.

- Ensures the `src/` directory is added to `sys.path` so that modules such as `train`, `infer`, `data`, etc. can be imported in a flat layout.
- This configuration is applied automatically whenever `pytest` is run in this directory.

---

### `test_config_and_imports.py`

Light-weight sanity checks for project layout and imports.

- Verifies that `src/` is on `sys.path`.
- Attempts to import a small set of core modules (`train`, `eval`, `infer`, `data`, `transforms`) and asserts that at least one of them is importable.
- These tests are intended to catch basic configuration or packaging issues early.

---

### `test_cli_parse_only.py`

Smoke tests for the command-line interfaces without running full training.

- Runs `python -m src.train`, `python -m src.eval` and `python -m src.infer` with lightweight argument combinations such as `--max-epochs 1 --dry-run` or `--help`.
- Confirms that each CLI parses arguments correctly and exits with status code 0.
- Uses the repository root on `PYTHONPATH` so that `src` is importable as a package namespace.

---

### `test_project_smoke.py`

High-level smoke tests for the repository.

- Checks that at least one of the following exists:
  - `src/train.py`, `src/infer.py`, `src/eval.py` (invoked as `python -m src.<name> --help`), or
  - Top-level scripts `train.py`, `infer.py`, `eval.py`.
- Ensures each script responds successfully to the `--help` flag.
- Confirms that a demo entrypoint `app.py` exists at the repository root and that the `streamlit` package is importable.

---

### `test_data_contract.py`

Contract tests for data and transform factories in a flat `src/` layout.

- Searches for a module named one of `data`, `transforms`, `dataset`, `datasets`.
- Within that module, searches for a transform factory with one of the following names:
  - `build_transforms`, `get_transforms`, `create_transforms`, `make_transforms`.
- Verifies that:
  - The factory is callable.
  - It can be invoked with minimal arguments.
  - It returns either:
    - a single callable-like object, or
    - a sequence / mapping of callable-like objects.
- Scans the same set of modules for dataset-like constructors or factories.
  - Ensures at least one has a reasonable signature for typical dataset arguments (e.g. `root`, `train`, `split`, `download`).

These tests enforce a flexible but explicit contract for the data-loading API.

---

### `test_data_runtime.py`

End-to-end tests for the data pipeline on a minimal synthetic Food-101-like layout.

- Builds a tiny data tree under a temporary directory:

  ```text
  <tmp>/food-101/
      images/<class>/<file>.jpg
      meta/train.txt
      meta/test.txt
  ```

- Uses real JPEG files written on the fly to exercise:
  - `read_splits` (parsing `train.txt` and `test.txt`),
  - `build_transforms` (creating training and validation transforms),
  - `make_datasets` (building `Dataset` objects and class mappings).
- Asserts that:
  - The number of samples in train and validation datasets matches the split files.
  - The set of discovered classes is correct.
  - A sample retrieved from the training dataset is a `torch.Tensor` with the expected spatial dimensions.

---

### `test_infer_api_contract.py`

Contract tests for the inference API in a flat layout.

- Searches for a callable named one of:
  - `infer`, `inference`, `predict`, `predict_image`, `run_inference`
- Looks for such callables inside modules:
  - `infer`, `utils`, `inference`, `api`, `service`
- Accepts either:
  - A public callable in one of those modules, or
  - The presence of a module `infer` (e.g. `src/infer.py` acting as a CLI).
- If a callable is found, verifies that calling it with clearly invalid input (such as `None`) raises an `Exception`, indicating basic input validation.

---

### `test_infer_runtime.py`

Runtime tests for the inference utilities using a tiny dummy model.

- Defines a minimal `DummyNet` subclass of `nn.Module` that:
  - Takes input of shape `(B, 3, H, W)`,
  - Produces logits of shape `(B, num_classes)` by applying a `1x1` convolution and spatial averaging.
- Exercises the following utilities from `utils`:
  - `PREPROCESS`: the standard preprocessing transform for PIL images.
  - `predict_pil`: runs a model on a PIL image and returns a list of `(index, probability)` pairs.
  - `topk_labels`: maps indices and probabilities to `(class_label, probability)` pairs given a list of class names.
- Verifies that:
  - `predict_pil` returns a list of the requested length `topk`.
  - Indices are integers, probabilities are floats in `[0, 1]`.
  - Probabilities are sorted in descending order.
  - `topk_labels` correctly maps indices to class names.

---

### `test_metrics_correctness.py`

Correctness tests for metric helpers in the training engine.

- Uses small, hand-crafted examples to validate:
  - `accuracy_topk` from `engine`, which returns counts of correct predictions for various top-k values.
  - `_to_prob_matrix`, which converts logits or probability-like arrays into a well-formed probability matrix.
- For `accuracy_topk`:
  - Constructs logits and targets where the top-1 and top-2 correctness can be computed exactly.
  - Asserts that the returned counts match the expected values.
- For `_to_prob_matrix`:
  - Checks that applying it to logits reproduces the result of a manual softmax computation.
  - Verifies that input rows that already represent valid probabilities are preserved (up to numerical noise) and still sum to 1.

---

### `test_train_happy_path.py`

End-to-end “happy path” test for the training CLI on a tiny synthetic dataset.

- Creates a minimal Food-101-like dataset under a temporary directory with:
  - Two classes,
  - One training and one test image per class.
- Writes a temporary YAML configuration (`train_tiny.yaml`) with:
  - A simple architecture identifier (`tiny_cnn`),
  - `pretrained` flag,
  - Small batch size and number of epochs,
  - CPU device,
  - `mlflow` disabled for faster test runtime.
- Invokes:

  ```bash
  python -m src.train --config <path_to_train_tiny.yaml>
  ```

  with `PYTHONPATH` set so that `src` is importable.
- Asserts that:
  - The process exits with status code 0.
  - An `artifacts/` directory is created in the repository root, indicating that training ran and produced outputs (such as checkpoints or metrics).

---

## Conventions and assumptions

The tests are written with the following conventions in mind:

- **Flat `src/` layout**  
  Core modules like `train.py`, `eval.py`, `infer.py`, `data.py`, `engine.py`, `utils.py` live directly under the `src/` directory and are importable as `train`, `eval`, `infer`, `data`, etc. when `src` is on `sys.path`.

- **Food-101–style dataset structure**  
  Where data is involved, tests build synthetic layouts that mimic the Food-101 directory structure, including `images/` and `meta/` folders with `train.txt` and `test.txt`.

- **Minimal external dependencies during tests**  
  Tests rely on small, synthetic datasets and short runs to keep execution fast, while still exercising the key code paths of data loading, training, inference and metrics.

This README belongs in the `tests/` directory and documents the intent and coverage of the test suite in that location.
