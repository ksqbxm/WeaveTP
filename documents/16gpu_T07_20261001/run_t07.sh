#!/usr/bin/env bash
# T07 static/CPU acceptance only. No SSH, torchrun, installs, or model reads.
set -euo pipefail
EXPECTED_COMMIT=${1:?Pass the published full T07 commit SHA}
: "${WORK:?Source env.sh first}" "${REPO:?}" "${CODE_DIR:?}" "${NODE1_CODE_DIR:?}"
: "${NODE0_PYTHON:?}" "${NODE1_PYTHON:?}" "${MASTER_ADDR:?}" "${NODE1_SSH:?}" "${MASTER_PORT:?}"
: "${CHECKPOINT:?Set the verified checkpoint path}"
T07_REPO=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
test "$T07_REPO" = "$REPO"
T07_CODE="$REPO/code/current"
if [[ "$CODE_DIR" != "$T07_CODE" ]]; then
    echo 'STOP: CODE_DIR must equal $REPO/code/current' >&2
    exit 1
fi
T07_WORK="$WORK"
case "$(hostname -s | tr '[:upper:]' '[:lower:]')" in
    sl3060) test "${NODE_RANK:?}" = 0
        test "${PYTHON:?}" = "$NODE0_PYTHON" ;;
    sl3061) test "${NODE_RANK:?}" = 1
        test "${PYTHON:?}" = "$NODE1_PYTHON" ;;
    *) echo 'STOP: expected SL3060 or SL3061' >&2; exit 1 ;;
esac
test -x "$PYTHON"
test "$(git -C "$T07_REPO" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$T07_REPO" status --porcelain --untracked-files=all)"
case "$(realpath "$T07_WORK")" in /data/*) ;; *) exit 1 ;; esac
case "$(realpath "$T07_REPO")" in "$T07_WORK"/*) ;; *) exit 1 ;; esac
df -h /data
T07_RUN="$T07_WORK/acceptance/t07_$(hostname -s)_$(date +%Y%m%dT%H%M%S)_$$"
mkdir -p "$T07_WORK/acceptance" "$T07_WORK/reports"
mkdir -m 700 "$T07_RUN"
finish() {
    rc=$?
    trap - EXIT
    printf '\nT07 host=%s commit=%s exit=%s evidence=%s (static/CPU only)\n' \
        "$(hostname -s)" "$EXPECTED_COMMIT" "$rc" "$T07_RUN" >> "$T07_WORK/reports/trial_report.md"
    printf 'T07_RESULT host=%s exit=%s evidence=%s\n' "$(hostname -s)" "$rc" "$T07_RUN"
    exit "$rc"
}
trap finish EXIT
exec > >(tee "$T07_RUN/run.log") 2>&1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 CUDA_VISIBLE_DEVICES=""
for key in TMPDIR TMP TEMP XDG_CACHE_HOME CUDA_CACHE_PATH TORCH_HOME TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR HF_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE PIP_CACHE_DIR; do
    mkdir "$T07_RUN/$key"
    export "$key=$T07_RUN/$key"
done
cd "$T07_RUN"
printf 'CODE_OK commit=%s GIT_CLEAN=1\n' "$EXPECTED_COMMIT"
"$PYTHON" -B -X utf8 - "$T07_REPO" "$T07_RUN" <<'PY'
import hashlib
import json
import pathlib
import subprocess
import sys
repo, out = map(pathlib.Path, sys.argv[1:])
manifest = json.loads((repo / 'documents/16gpu_T07_20261001/source_manifest.json').read_bytes())
for row in manifest['files']:
    path = repo / row['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], path
    if path.suffix == '.py':
        compile(path.read_bytes(), str(path), 'exec')
    elif path.suffix == '.sh':
        subprocess.run(['bash', '-n', str(path)], check=True)
summary = repo / 'code/current/tools/resharding/summarize_weavetp_formal.py'
assert hashlib.sha256(summary.read_bytes()).hexdigest() == manifest['t03_summarizer_sha256']
(out / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(f"STATIC_OK files={len(manifest['files'])} hash/compile/bash-n")
print('T03_SUMMARIZER_UNCHANGED (prior historical acceptance; not rerun here)')
PY
"$PYTHON" -B -X utf8 "$T07_CODE/tests/server_tests/weavetp_16gpu/test_cpu.py" --output-dir "$T07_RUN/cpu"
"$PYTHON" -B -X utf8 - "$T07_RUN/cpu" <<'PY'
import json
import pathlib
import re
import sys
out = pathlib.Path(sys.argv[1])
result = json.loads((out / 'acceptance.json').read_bytes())
assert result['status'] == 'passed', result
expected = [28, 26, 18, 11, 4, 25]
for name, count in zip(result['details']['suites'], expected, strict=True):
    log = json.loads((out / (name.removesuffix('.py') + '.json')).read_bytes())
    assert log['exit_code'] == 0 and re.search(rf'Ran {count} tests\b', log['stderr']), name
print('CPU_OK counts=28/26/18/11/4/25 total=112')
PY
"$PYTHON" -B -X utf8 "$T07_REPO/documents/16gpu_T06_T07_review_20261001/reproduce_findings.py"
echo 'REVIEW_OK tests=3 total_cpu_tests=115'
# These paths are labels only: neither dry-run creates its output/profile path.
OUT_DIR="$T07_RUN/profile_not_executed" bash "$T07_CODE/tools/resharding/run_weavetp_16gpu_profile.sh" \
    --dry-run > "$T07_RUN/profile_dry_run.txt"
test ! -e "$T07_RUN/profile_not_executed"
echo 'PROFILE_DRY_RUN_OK (no GPU started)'
if [[ "$NODE_RANK" = 0 ]]; then
    NNODES=2 CODE_DIR="$T07_CODE" NODE1_CODE_DIR="$NODE1_CODE_DIR" NODE1_PYTHON="$NODE1_PYTHON" \
        ROOT_OUT="$T07_RUN/formal_not_executed" PROFILE="$T07_RUN/profile_not_executed/profile.json" \
        CHECKPOINT="$CHECKPOINT" \
        bash "$T07_CODE/tools/resharding/run_deepseek_v2_lite_directional_wave_compare.sh" \
        --dry-run > "$T07_RUN/compare_dry_run.json"
    "$PYTHON" -B -X utf8 - "$T07_RUN/compare_dry_run.json" "$NODE1_PYTHON" <<'PY'
import json
import sys
rows = json.load(open(sys.argv[1], encoding='utf-8'))
assert [(r['request']['repeat'], r['request']['case']) for r in rows] == [
    (1, 'fixed'), (1, 'directional'), (1, 'weavetp'),
    (2, 'directional'), (2, 'weavetp'), (2, 'fixed'),
    (3, 'weavetp'), (3, 'fixed'), (3, 'directional')]
for row in rows:
    request = row['request']
    assert len(row['node_commands']) == 2
    assert request['env']['NCCL_DEBUG'] == 'WARN'
    assert request['env']['GLOBAL_BATCH_SIZE'] == '8'
    assert request['nodes'][1]['python'] == sys.argv[2]
print('COMPARE_DRY_RUN_OK cases=9 nodes=2 WARN global_batch=8')
PY
    test ! -e "$T07_RUN/formal_not_executed"
fi
git -C "$T07_REPO" diff --check
test -z "$(git -C "$T07_REPO" status --porcelain --untracked-files=all)"
echo 'T07_STATIC_CPU_OK GIT_CLEAN=1 GPU_STARTED=0'
