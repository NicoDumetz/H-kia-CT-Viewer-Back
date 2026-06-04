#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

cd "$ROOT_DIR"
mkdir -p storage/studies

python -m uvicorn app.main:app --reload --host "$HOST" --port "$PORT"
