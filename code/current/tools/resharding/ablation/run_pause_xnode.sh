#!/usr/bin/env bash
# pause entry, cross-machine point. Run on SL3061 with NODE_RANK=0 (master), then on SL3060 with NODE_RANK=1.
MODE=pause
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
use_xnode
preflight 1
xnode
