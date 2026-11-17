FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

LABEL org.opencontainers.image.source=https://github.com/manascb1344/zonos-api
LABEL org.opencontainers.image.description="Zonos API with GPU support"
LABEL org.opencontainers.image.licenses=Apache-2.0

WORKDIR /app

# System packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    espeak-ng \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir uv

# Clone Zonos into /app/Zonos and leave its own pyproject.toml intact
RUN git clone --depth 1 https://github.com/Zyphra/Zonos.git /app/Zonos

# Copy dependency specs, README for hatchling metadata, and application code
COPY requirements.txt pyproject.toml README.md /app/
COPY app/ /app/app/

RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system -r /app/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system --no-deps -e /app

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/audio /app/.cache/huggingface \
    && chown -R appuser:appuser /app

USER appuser

ENV PYTHONPATH=/app:/app/Zonos \
    HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
