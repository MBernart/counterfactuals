FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY ./explainers/growing_spheres/req.txt ./req.txt
RUN pip install --no-cache-dir --no-compile --target /opt/pip-packages -r req.txt

COPY ./src ./src
RUN cd ./src && pip install --no-cache-dir --no-compile --target /opt/pip-packages .

RUN find /opt/pip-packages -type d -name "__pycache__" -exec rm -rf {} + \
 && find /opt/pip-packages -type d -name "tests" -exec rm -rf {} + \
 && rm -rf /opt/pip-packages/pip* /opt/pip-packages/setuptools* /opt/pip-packages/wheel* \
 && find /opt/pip-packages -type f -name "*.pyc" -delete

FROM python:3.13-slim

ENV PYTHONPATH=/opt/pip-packages:/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgomp1 libgfortran5 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/pip-packages /opt/pip-packages

RUN groupadd -r appgroup && useradd -r -g appgroup appuser

WORKDIR /app
COPY --chown=appuser:appgroup ./explainers/growing_spheres/ ./

USER appuser
EXPOSE 8000
CMD ["python", "main.py"]