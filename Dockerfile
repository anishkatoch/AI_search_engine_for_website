# syntax=docker/dockerfile:1

# ============================================================================
# Stage 1 — build the React frontend into static files (frontend/dist).
# ============================================================================
FROM node:20-bookworm-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ============================================================================
# Stage 2 — Python runtime that serves both the API and the built UI.
# uv + Python preinstalled for fast, reproducible installs from uv.lock.
# ============================================================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Install dependencies first (cached unless pyproject.toml / uv.lock change).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# App code, config, data + the prebuilt frontend from stage 1.
# NOTE: .env (secrets) is intentionally NOT copied — pass keys at runtime.
COPY app/ ./app/
COPY config.yaml amazon_products.csv ./
COPY --from=frontend /frontend/dist ./frontend/dist

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8005

# One process serves the React UI at / and the API at /api/*.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]
