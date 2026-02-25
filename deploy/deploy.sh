#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/deploy/docker-compose.prod.yml"

run_local() {
  echo "Docker not found. Falling back to local backend deployment."
  if ! python - <<'PY'
import flask
import flask_cors
import numpy
import pandas
PY
  then
    echo "Error: local fallback requires python packages: flask, flask-cors, numpy, pandas" >&2
    echo "Install them with: pip install flask flask-cors numpy pandas" >&2
    exit 1
  fi

  echo "Starting local backend on http://localhost:5000"
  echo "Open frontend at: http://localhost:5000"
  exec python "$ROOT_DIR/flask_app.py"
}

if ! command -v docker >/dev/null 2>&1; then
  run_local
fi

if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD=(docker-compose)
else
  run_local
fi

"${COMPOSE_CMD[@]}" -f "$COMPOSE_FILE" up -d --build
"${COMPOSE_CMD[@]}" -f "$COMPOSE_FILE" ps

echo "Deployment completed. Frontend is available on http://localhost:${FRONTEND_PORT:-80}"
