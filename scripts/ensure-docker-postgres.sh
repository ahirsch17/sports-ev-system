#!/usr/bin/env bash
# macOS/Linux counterpart of ensure-docker-postgres.bat: start Docker + Postgres, apply migrations.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x ".venv/bin/python" ]; then
  echo "Virtual environment not found at .venv" >&2
  exit 1
fi
PY=.venv/bin/python

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker CLI not found. Install Docker Desktop (or OrbStack/Colima)." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  if [ "$(uname)" = "Darwin" ]; then
    echo "Docker is not running. Starting Docker Desktop..."
    open -a Docker || true
  fi
  for _ in $(seq 1 40); do
    docker info >/dev/null 2>&1 && break
    sleep 3
  done
  docker info >/dev/null 2>&1 || { echo "Docker did not start" >&2; exit 1; }
fi

echo "Starting Postgres if needed..."
docker compose up -d

echo "Waiting for Postgres on port 5433..."
for _ in $(seq 1 30); do
  if "$PY" -c "from sports_ev.config import get_settings; from sports_ev.db.session import create_db_engine; from sqlalchemy import text; e=create_db_engine(get_settings()); c=e.connect(); c.execute(text('select 1')); c.close()" >/dev/null 2>&1; then
    echo "Postgres is ready."
    "$PY" -m sports_ev.cli migrate-db >/dev/null 2>&1 || true
    exit 0
  fi
  sleep 2
done
echo "Postgres not responding on port 5433" >&2
exit 1
