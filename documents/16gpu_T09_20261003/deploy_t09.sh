#!/usr/bin/env bash
# Run manually on SL3061 first, then SL3060. CPU checks only; no launch.
set -euo pipefail
target_input=${TARGET:?Set TARGET to the full released commit SHA}
source /data/ubuntu/lxh/weavetp/env.sh
test -z "$(git -C "$REPO" status --porcelain --untracked-files=all)"
case "$(hostname -s)" in
  SL3060|SL3061) ;;
  *) echo 'STOP: expected SL3060/SL3061' >&2; exit 1 ;;
esac
git -C "$REPO" checkout --detach "$target_input"
test "$(git -C "$REPO" rev-parse HEAD)" = "$target_input"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=all)"
"$PYTHON" -B - "$WORK/env.sh" "$target_input" <<'PY'
import re
import sys
from pathlib import Path
path = Path(sys.argv[1])
raw = path.read_text()
updated, count = re.subn(r'(?m)^(\s*(?:export\s+)?WEAVETP_COMMIT=).*$',
                        lambda m: m[1] + "'" + sys.argv[2] + "'", raw)
if count != 1:
    raise SystemExit('STOP: expected exactly one WEAVETP_COMMIT assignment')
path.write_text(updated)
PY
source "$WORK/env.sh"
test "$WEAVETP_COMMIT" = "$target_input"
cd "$REPO"
sha256sum -c documents/16gpu_T09_20261003/t09.sha256
"$PYTHON" -B -X utf8 documents/16gpu_T09_20261003/check_cpu.py \
  --output-dir "$WORK/acceptance/t09_cpu_$(hostname -s)_$(date -u +%Y%m%dT%H%M%SZ)_$$"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=all)"
echo "T09_CODE_READY host=$(hostname -s) commit=$target_input GPU_STARTED=0"
