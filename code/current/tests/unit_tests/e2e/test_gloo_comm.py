"""GlooComm over two CPU processes: the leader's decision reaches the peer, all_ready is a global MIN."""

import json
import os
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[3]
WORKER = '''
import json, sys
import torch.distributed as dist
from tools.resharding.e2e import controller as ctl
from tools.resharding.e2e.live_runtime import GlooComm
rank, init_file, out = int(sys.argv[1]), sys.argv[2].replace(chr(92), "/"), sys.argv[3]
dist.init_process_group("gloo", init_method="file:///" + init_file.lstrip("/"), rank=rank, world_size=2)
comm = GlooComm("EDP0", (0, 1), dist.new_group([0, 1]))
decision = ctl.Decision(for_step=7, admits=[(3, 0, 5)], need_prefill_forward=True) if comm.leader else None
received = comm.broadcast(decision)
results = {"leader": comm.leader, "for_step": received.for_step, "admits": received.admits,
           "need_prefill_forward": received.need_prefill_forward,
           "all_ok": comm.all_ready(True), "one_failed": comm.all_ready(rank == 0)}
dist.destroy_process_group()
open(out, "w").write(json.dumps(results))
'''


def test_gloo_comm_broadcasts_the_leader_decision_and_votes_min(tmp_path):
    init = tmp_path / 'init'
    env = {**os.environ, 'PYTHONPATH': str(CODE_ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    children = [subprocess.Popen([sys.executable, '-B', '-c', WORKER, str(rank), str(init), str(tmp_path / f'rank{rank}.json')],
                                 env=env, cwd=CODE_ROOT) for rank in (0, 1)]
    for child in children:
        assert child.wait(timeout=180) == 0
    results = [json.loads((tmp_path / f'rank{rank}.json').read_text()) for rank in (0, 1)]
    assert [r['leader'] for r in results] == [True, False]
    assert all(r['for_step'] == 7 and r['admits'] == [[3, 0, 5]] and r['need_prefill_forward'] for r in results)
    assert all(r['all_ok'] is True and r['one_failed'] is False for r in results)
