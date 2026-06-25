# ── Satellite Change Detection — Cloud Run Image ──────────────────────────────
# Build context: project root (one level above cloud_deployment/)
#   docker build -f cloud_deployment/Dockerfile -t scd-app .
#   docker run -p 8080:8080 scd-app

# Stage 1: Python + GDAL base
FROM python:3.11-slim AS base

# System deps for GDAL / rasterio / geopandas
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV GDAL_VERSION=3.6.4 \
    CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# Stage 2a: PyTorch CUDA — installed separately so the index-url is explicit
# and this heavy layer (~2.5 GB) is cached independently of other deps.
RUN pip install --no-cache-dir \
    torch==2.4.1+cu124 \
    torchvision==0.19.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Stage 2b: Remaining Python deps (lighter packages from PyPI)
COPY cloud_deployment/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Copy project source
# Project utilities and methods (from project root)
COPY methods/ ./methods/
COPY utils/   ./utils/

# Web app
COPY cloud_deployment/app/ ./app/

# AOI file (optional, used for map centering)
COPY aoi.geojson ./aoi.geojson

# ── Runtime ───────────────────────────────────────────────────────────────────
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

# Cloud Run sets PORT env var; uvicorn reads it
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
