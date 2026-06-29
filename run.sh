#!/usr/bin/env bash
#
# Zero-setup launcher. On a fresh machine (e.g. a new AWS EC2 instance) this:
#   1. installs uv if it's missing,
#   2. syncs the locked Python dependencies into ./.venv,
#   3. builds the React frontend (if Node is available and dist is missing),
#   4. starts ONE server that serves the UI at /  and the API at /api/*.
#
#   Open http://localhost:8005
#
# Usage:  ./run.sh

set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8005}"

# 1) Ensure uv is available.
if ! command -v uv >/dev/null 2>&1; then
  echo "==> uv not found, installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# 2) Install the exact locked dependencies (creates .venv on first run).
echo "==> Syncing Python dependencies..."
uv sync --frozen

# 3) Build the frontend if needed.
if [ ! -f frontend/dist/index.html ]; then
  if command -v npm >/dev/null 2>&1; then
    echo "==> Building React frontend..."
    (cd frontend && npm install && npm run build)
  else
    echo "!! Node/npm not found — the UI won't be served until the frontend is built."
    echo "   Install Node, then run: (cd frontend && npm install && npm run build)"
  fi
fi

# 4) Run the single server (UI + API).
echo "==> Starting server on http://localhost:${PORT}  (API docs: /docs)"
exec uv run uvicorn app.main:app --host 0.0.0.0 --port "${PORT}"
