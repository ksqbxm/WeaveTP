"""TEST DOUBLE ONLY: installed in a disposable test code copy, never imports torch.

Its temporary basename matches the production worker so /proc tagging and real
signals exercise the production cleanup path. The result is synthetic evidence.
"""

import os
import sys
import time
from pathlib import Path

out = Path(sys.argv[sys.argv.index("--out-dir") + 1])
rank = int(os.environ["NODE_RANK"])
scenario = os.environ["SERVER_TEST_SCENARIO"]
(out / f"cpu_worker_started.node{rank}").write_text(str(os.getpid()))
if scenario in ("node1-failure", "controller-interrupt"):
    if rank == 1 and scenario == "node1-failure":
        time.sleep(1)
        raise SystemExit(7)
    time.sleep(30)  # Finite even if production cleanup fails; never occupies a GPU.
else:
    time.sleep(1)
if rank == 0:
    fixture = Path(__file__).resolve().parents[2] / "fixture_result.json"
    (out / "result.json").write_bytes(fixture.read_bytes())
