# syntax=docker/dockerfile:1

# ---- builder: resolve deps into a venv we can copy ----
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY pyproject.toml ./
RUN uv venv /opt/venv && VIRTUAL_ENV=/opt/venv uv pip install -r pyproject.toml

# ---- runtime: no build toolchain, non-root ----
FROM python:3.11-slim AS runtime
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
COPY --chown=app:app src/ ./src/
COPY --chown=app:app db/ ./db/
ENV PYTHONPATH=/app/src
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=40s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"
CMD ["uvicorn", "groundtruth.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
