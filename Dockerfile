# syntax=docker/dockerfile:1

# ---------- frontend build ----------
FROM node:22-alpine AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---------- python base (CPU PyTorch) ----------
FROM python:3.12-slim AS backend-base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/data \
    CHECKPOINT_DIR=/checkpoints \
    STATIC_DIR=/app/static
WORKDIR /app
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
COPY backend/requirements.txt ./
RUN pip install -r requirements.txt

# ---------- dev: source is bind-mounted, data lives in a named volume ----------
FROM backend-base AS dev
COPY backend/requirements-dev.txt ./
RUN pip install -r requirements-dev.txt
EXPOSE 8000
CMD ["sh", "-c", "python -m app.data --prepare && python -m app.pretrain --if-missing && exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --timeout-graceful-shutdown 2"]

# ---------- production: one self-contained image ----------
FROM backend-base AS runtime
COPY backend/app ./app
# Bake MNIST (16x16 cache) and the pretrained checkpoint into the image so it runs offline.
RUN python -m app.data --prepare \
    && rm -rf /data/MNIST \
    && python -m app.pretrain
COPY --from=frontend-build /frontend/dist ./static
RUN useradd --create-home app && mkdir -p /checkpoints && chown app /checkpoints
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/model/info')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
