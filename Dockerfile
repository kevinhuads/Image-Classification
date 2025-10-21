# Use Python 3.11 slim image
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (adjust if you need OpenCV, etc.)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libglib2.0-0 libsm6 libxrender1 libxext6 \
#   && rm -rf /var/lib/apt/lists/*

# Install only requirements first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the source code
COPY . /app

# If your project is a package, install it:
# RUN pip install -e .

# Expose a port if you run a web service (uncomment and set correctly)
# EXPOSE 8000

# Set the default command (adjust to how you run your app)
# Example: python -m your_package.cli or python src/main.py
CMD ["python", "-c", "print('Container built successfully. Replace CMD with your app entrypoint.')"]