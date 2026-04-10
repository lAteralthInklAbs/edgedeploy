# Multi-stage Dockerfile for EdgeDeploy
# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install package and dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir . \
    && pip wheel --no-cache-dir --wheel-dir=/wheels .

# Stage 2: Runtime
FROM python:3.11-slim as runtime

# Create non-root user
RUN groupadd --gid 1000 edgedeploy \
    && useradd --uid 1000 --gid 1000 --create-home edgedeploy

WORKDIR /app

# Copy wheels from builder
COPY --from=builder /wheels /wheels
COPY --from=builder /app/src /app/src
COPY pyproject.toml .

# Install from wheels
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir /wheels/*.whl \
    && rm -rf /wheels

# Copy scripts and fixtures for evaluation
COPY scripts/ scripts/
COPY fixtures/ fixtures/
COPY configs/ configs/

# Create results directory
RUN mkdir -p /app/results && chown -R edgedeploy:edgedeploy /app

# Switch to non-root user
USER edgedeploy

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.drift import DriftEnsemble; print('healthy')" || exit 1

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "-c", "from src import __version__; print(f'EdgeDeploy v{__version__}')"]


