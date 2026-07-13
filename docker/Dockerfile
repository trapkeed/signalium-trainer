FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

LABEL maintainer="Signalium <engineering@signalium.io>"
LABEL description="Signalium Trainer — GPU training container"

ARG PYTHON_VERSION=3.12
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python3-pip \
        git \
        curl \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --break-system-packages --no-cache-dir -e ".[cuda]"

COPY . .

RUN pip install --break-system-packages --no-cache-dir -e "."

VOLUME ["/app/data", "/app/checkpoints", "/app/exports", "/app/logs", "/app/mlruns"]

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TORCH_HOME=/app/.torch
ENV MLFLOW_TRACKING_URI=/app/mlruns

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; import trainer; print('OK')" || exit 1

ENTRYPOINT ["trainer"]
CMD ["doctor"]
