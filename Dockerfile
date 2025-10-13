# syntax=docker/dockerfile:1

# ========= Base image =========
FROM python:3.11-slim

# Avoid interactive prompts, speed up pip
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ========= System deps =========
# libgl1 / libglib2.0-0 are needed for OpenCV/torchvision image ops at runtime
# build-essential is *not* required for the pinned wheels, so we skip it to keep the image small
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ========= Workdir =========
WORKDIR /app

# Copy only requirement file first for better layer caching
COPY requirements.txt ./

# -------- Install PyTorch CPU wheels first (match pinned versions) --------
# We install torch/torchvision from the official CPU wheel index to avoid pulling CUDA.
RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.8.0 torchvision==0.23.0

# -------- Install the rest of the requirements --------
# Exclude torch and torchvision (already installed) to prevent duplicates/conflicts.
RUN awk 'BEGIN{IGNORECASE=1} !/^torch(|vision)==/' requirements.txt > requirements.nopytorch.txt && \
    python -m pip install --no-cache-dir -r requirements.nopytorch.txt && \
    rm -f requirements.nopytorch.txt

# ========= App code =========
COPY . .

# -------- Patch Windows-only checkpoint paths to be configurable --------
# app.py: replace hard-coded CKPT with an env-driven default (/models/headonly_food101.pth)
# infer.py: update the default value of --ckpt to the same Linux path (still overrideable by CLI).
RUN python - <<'PY'\n\
import io, os, re\n\
# app.py\n\
p = 'app.py'\n\
s = open(p, 'r', encoding='utf-8').read()\n\
# ensure os is imported\n\
if 'import os' not in s:\n\
    s = s.replace('import streamlit as st', 'import streamlit as st\\nimport os')\n\
# replace CKPT assignment line\n\
s = re.sub(r\"^CKPT\\s*=\\s*r?['\\\"](.+?)['\\\"]\", \"CKPT = os.getenv('CKPT_PATH', '/models/headonly_food101.pth')\", s, flags=re.M)\n\
open(p, 'w', encoding='utf-8').write(s)\n\
# infer.py\n\
p = 'infer.py'\n\
s = open(p, 'r', encoding='utf-8').read()\n\
s = s.replace(r\"default=r\\\"D:\\\\Python\\\\food-101\\\\headonly_food101.pth\\\"\", \"default='/models/headonly_food101.pth'\")\n\
open(p, 'w', encoding='utf-8').write(s)\n\
print('Patched app.py and infer.py for portable checkpoint paths.')\n\
PY

# Where the model checkpoint will be mounted (bind mount or COPY during build)
# You can COPY your checkpoint into the image by uncommenting the next line, or bind-mount at runtime.
# COPY headonly_food101.pth /models/headonly_food101.pth
RUN mkdir -p /models

# Default: expose Streamlit port
EXPOSE 8501

# Healthcheck: basic TCP check on Streamlit port
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python - <<'PY'\n\
import socket\n\
s=socket.socket()\n\
try:\n\
    s.settimeout(2)\n\
    s.connect(('127.0.0.1',8501))\n\
    print('ok')\n\
    raise SystemExit(0)\n\
except Exception:\n\
    raise SystemExit(1)\n\
finally:\n\
    s.close()\n\
PY

# Runtime env for the checkpoint (can be overridden with -e CKPT_PATH=...)
ENV CKPT_PATH=/models/headonly_food101.pth

# Default command runs the Streamlit UI.
CMD [\"streamlit\", \"run\", \"app.py\", \"--server.port\", \"8501\", \"--server.address\", \"0.0.0.0\"]\n