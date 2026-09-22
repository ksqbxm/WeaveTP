#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
mkdir -p "$ROOT/logs" "$ROOT/results"
date --iso-8601=seconds > "$ROOT/results/firm-moe-smoke.started"
"$PYTHON" "$ROOT/implementation/firm_moe_smoke.py" \
  --output "$ROOT/results/firm-moe-smoke.json" \
  2>&1 | tee "$ROOT/logs/firm-moe-smoke.log"
date --iso-8601=seconds > "$ROOT/results/firm-moe-smoke.completed"
