#!/usr/bin/env bash
set -euo pipefail

# The old wrapper name remains as a compatibility entry point.
exec bash tools/resharding/run_live_moe_dual_objective_benchmark.sh "$@"
