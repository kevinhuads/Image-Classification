# Food-101 — ResNet-50 Classifier (Refactored)

## Overview
Compact pipeline for training, validating, and serving a ResNet-50 image classifier. Includes data loading, a training loop with mixed precision, inference utilities, and a small Streamlit demo.

## Repository
- `data.py` — dataset readers and transforms.  
- `model.py` — ResNet-50 builder (replace final layer for `num_classes`).  
- `engine.py` — train/validation loops, checkpointing, CSV logging.  
- `train.py` — training entrypoint.  
- `infer.py` — CLI inference.  
- `app.py` — Streamlit demo.  
- `utils.py` — model/load helpers.  
- `requirements.txt` — pinned dependencies.

## Quick start
1. Create environment and install:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Train:
   ```bash
   python train.py --data_folder /path/to/data --epochs 10 --batch_size 64 --pretrained
   ```

3. Inference (CLI):
   ```bash
   python infer.py /path/to/image.jpg --ckpt /path/to/refactored_best.pth --topk 5
   ```

4. Demo:
   ```bash
   streamlit run app.py
   ```

## Data layout
Expected structure:
```
DATA_FOLDER/
  images/<class>/<image>.jpg
  meta/train.txt   # lines like: class123/12345  (no .jpg)
  meta/test.txt
```

## Checkpoints
Saved checkpoint contains `epoch`, `model_state_dict`, `optimizer`, and `classes`. `utils.load_model_and_meta` reconstructs the model using the `classes` order.

## Notes
- `train.py` uses mixed precision and OneCycleLR by default.  
- The provided `requirements.txt` may contain CUDA-specific wheels; adjust for local CUDA or CPU-only environments.  
- Some scripts include Windows path defaults—update `CKPT` or pass `--ckpt` where required.
