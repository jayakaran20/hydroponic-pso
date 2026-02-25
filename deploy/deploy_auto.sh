#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/deploy/docker-compose.prod.yml"
LOCAL_PID_FILE="$ROOT_DIR/deploy/.local_backend.pid"

wait_for_health() {
  local url="http://127.0.0.1:5000/api/health"
  for _ in {1..20}; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

start_with_docker() {
  local compose_cmd=()
  if docker compose version >/dev/null 2>&1; then
    compose_cmd=(docker compose)
  elif command -v docker-compose >/dev/null 2>&1; then
    compose_cmd=(docker-compose)
  else
    return 1
  fi

  "${compose_cmd[@]}" -f "$COMPOSE_FILE" up -d --build
  "${compose_cmd[@]}" -f "$COMPOSE_FILE" ps
  echo "✅ Docker deployment completed. Frontend: http://localhost:${FRONTEND_PORT:-80}"
}

start_local_fallback() {
  echo "⚠️ Docker not available. Falling back to local deployment (single Flask service for UI + API)."

  local py_bin="python3"

  if [[ "${SKIP_PIP_INSTALL:-0}" != "1" ]]; then
    if [[ ! -d "$ROOT_DIR/.venv" ]]; then
      python3 -m venv "$ROOT_DIR/.venv"
    fi
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.venv/bin/activate"
    py_bin="python"
    "$py_bin" -m pip install --upgrade pip >/dev/null
    "$py_bin" -m pip install -r "$ROOT_DIR/requirements.txt"
  fi

  if [[ -f "$LOCAL_PID_FILE" ]] && ! kill -0 "$(cat "$LOCAL_PID_FILE")" 2>/dev/null; then
    rm -f "$LOCAL_PID_FILE"
  fi

  if [[ -f "$LOCAL_PID_FILE" ]] && kill -0 "$(cat "$LOCAL_PID_FILE")" 2>/dev/null; then
    echo "Local backend already running with PID $(cat "$LOCAL_PID_FILE")."
  else
    rm -f "$ROOT_DIR/deploy/local_backend.log"

    pushd "$ROOT_DIR" >/dev/null
    nohup env FLASK_DEBUG=0 "$py_bin" flask_app.py > "$ROOT_DIR/deploy/local_backend.log" 2>&1 &
    local pid=$!
    popd >/dev/null
    echo "$pid" > "$LOCAL_PID_FILE"
    sleep 2
  fi

  if wait_for_health; then
    echo "✅ Local deployment completed. Frontend + API: http://localhost:5000"
  else
    echo "❌ Local deployment started but health check failed. See deploy/local_backend.log" >&2
    exit 1
  fi
}

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  if ! start_with_docker; then
    start_local_fallback
  fi
else
  start_local_fallback
fi
