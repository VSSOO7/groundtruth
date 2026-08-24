# syntax=docker/dockerfile:1

# ---- builder: resolve deps into a venv we can copy ----
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PROJECT_ENVIRONMENT=/opt/venv
# Copy the lockfile too: `uv sync --frozen` installs the exact versions CI
# tested. Installing from pyproject.toml alone re-resolves on every build, so
# the image could ship dependency versions no test ever ran against.
# The venv is built at its FINAL path (/opt/venv) because console-script
# shebangs bake in an absolute interpreter path -- building at /app/.venv and
# copying would leave `uvicorn` pointing at a directory the runtime lacks.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

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
