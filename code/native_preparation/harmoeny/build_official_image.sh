#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs"
IMAGE="harmoeny:ee50b188"
mkdir -p "$LOG_DIR"

docker build \
  --network host \
  --build-arg USER_ID="$(id -u)" \
  --build-arg GROUP_ID="$(id -g)" \
  --build-arg USER_NAME="$(whoami)" \
  --tag "$IMAGE" \
  "$ROOT/upstream" 2>&1 | tee "$LOG_DIR/docker-build-official.log"
