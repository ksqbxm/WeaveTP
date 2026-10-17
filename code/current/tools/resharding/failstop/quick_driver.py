"""Five-minute opt-in correctness launcher, fixed commit, peer handshake and restore."""

import argparse
import json
import math
import re
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

BASE = 'f7d555998b45756998465924d733fd70e553d6b4'


def send(stream, value):
    stream.write((json.dumps(value) + '\n').encode())
    stream.flush()


def receive(stream):
    data = stream.readline(65536)
    if not data or not data.endswith(b'\n'):
        raise RuntimeError('peer disconnected or invalid handshake')
    return json.loads(data)


def rendezvous(node, sha, port=29839, step=2):
    """SL3060 owns the run ID; neither node uses an independent suite timestamp."""
    handshake_deadline = time.monotonic() + 60
    if node == 'SL3060':
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('10.60.14.1', port))
            listener.listen(1)
            listener.settimeout(60)
            conn, _ = listener.accept()
        conn.settimeout(max(1, handshake_deadline - time.monotonic()))
        stream = conn.makefile('rwb')
        request = receive(stream)
        if request != {'node': 'SL3061', 'sha': sha, 'step': step}:
            raise RuntimeError('peer step/SHA/node mismatch')
        suite = f's{step}q_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
        send(stream, {'suite': suite, 'sha': sha})
    else:
        deadline = time.monotonic() + 60
        while True:
            try:
                conn = socket.create_connection(('10.60.14.1', port), timeout=2)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('waiting for SL3060 rendezvous exceeded 60 seconds')
                time.sleep(0.25)
        conn.settimeout(max(1, handshake_deadline - time.monotonic()))
        stream = conn.makefile('rwb')
        send(stream, {'node': node, 'sha': sha, 'step': step})
        reply = receive(stream)
        if reply['sha'] != sha or not re.fullmatch(rf's{step}q_[A-Za-z0-9_]+', reply['suite']):
            raise RuntimeError('invalid peer run ID/SHA')
        suite = reply['suite']
    return conn, stream, suite


def tee(stream, path, tail, keep=200):
    """Mirror runner output to nohup.log and runner.log; keep the tail for the terminal error."""
    with open(path, 'ab') as output:
        for line in iter(stream.readline, b''):
            output.write(line)
            output.flush()
            text = line.decode(errors='replace')
            sys.stdout.write(text)
            sys.stdout.flush()
            tail.append(text.rstrip())
            del tail[:-keep]


def runner_error(tail):
    """The child's own exception line, e.g. 'RuntimeError: insufficient /data space: {...}'."""
    errors = [line for line in tail if re.match(r'\w+(Error|Exception|Interrupt)\b', line)]
    return errors[-1] if errors else (tail[-1] if tail else 'runner produced no output')


def main():
    import fcntl

    parser = argparse.ArgumentParser()
    parser.add_argument('--node', choices=('SL3060', 'SL3061'), required=True)
    parser.add_argument('--step', choices=(2, 3, 4, 5), type=int, default=2)
    parser.add_argument('--min-free-disk-gb', type=float, default=1.0,
                        help='quick correctness threshold for /data; formal acceptance keeps the runner default of 5')
    parser.add_argument('--sha', required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--log-dir', type=Path, required=True)
    args = parser.parse_args()
    started, deadline = time.monotonic(), time.monotonic() + 300
    run = args.log_dir
    conn = stream = child = None
    restore = False
    status = {'step': args.step, 'node': args.node, 'quick_mode': True, 'correctness_only': True,
              'shared_gpu_mode': True, 'timing_valid': False, 'accepted': False, 'result': 'FAIL'}
    # Steps share a checkout, so all quick steps must hold the same local lock.
    lock = (args.repo / '.failstop_quick.lock').open('w')

    def git(*argv):
        return subprocess.check_output(['git', '-C', str(args.repo), *argv], text=True, stderr=subprocess.STDOUT).strip()

    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not re.fullmatch('[0-9a-f]{40}', args.sha):
            raise RuntimeError('delivery must pin a full SHA')
        if git('status', '--porcelain', '--untracked-files=no'):
            raise RuntimeError('tracked repository files are dirty')
        git('cat-file', '-e', BASE + '^{commit}')
        restore = True
        git('checkout', '--detach', args.sha)
        if git('rev-parse', 'HEAD') != args.sha:
            raise RuntimeError('checkout identity mismatch')
        print('Waiting for peer: timeout=60s; total quick budget=300s', flush=True)
        status['peer_wait_start'] = time.monotonic()
        conn, stream, suite = rendezvous(args.node, args.sha, step=args.step)
        status['peer_wait_s'] = time.monotonic() - status['peer_wait_start']
        run = args.log_dir.parent.parent / suite
        status['suite'], status['run'] = suite, str(run)
        (args.log_dir / 'run_path.txt').write_text(str(run) + '\n')
        entry = args.repo / 'code/current/tools/resharding/failstop' / {
            2: 'run_topology.py', 3: 'run_sources.py', 4: 'run_async.py', 5: 'run_model.py'}[args.step]
        # Step 5 runs its node's layouts under the deterministic contract; the fault matrix is Steps 2-4.
        argv = [sys.executable, '-B', str(entry), '--node', args.node, '--suite-id', suite, '--output', str(run),
                *([] if args.step == 5 else ['--mode', 'both', '--rounds', '1', '--real-launches', '1']),
                '--quick', '--correctness-only', '--budget-seconds', str(max(1, deadline - time.monotonic() - 45)),
                '--min-free-disk-gb', str(args.min_free_disk_gb)]
        runner_log = args.log_dir / 'runner.log'
        child = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        tail = []
        copier = threading.Thread(target=tee, args=(child.stdout, runner_log, tail), daemon=True)
        copier.start()
        code = child.wait(timeout=max(1, deadline - time.monotonic() - 40))
        copier.join(timeout=10)
        if code:
            progress = run / args.node / 'progress.json'
            if args.step == 5:
                hint = f"inspect {run / args.node / 'error.txt'} and the <layout>/<role>/torchrun.log files"
            elif progress.exists():
                status['failed_cases'] = [r for r in json.loads(progress.read_text()) if r['status'] != 'passed']
                hint = 'inspect per-round worker_error/controller_result logs'
            else:
                hint = 'no case started; the runner failed during preflight'
            status['runner_error'] = runner_error(tail)
            raise RuntimeError(f"runner exited {code}: {status['runner_error']} ({hint}; full output: {runner_log})")
        summary = json.loads((run / args.node / 'summary.json').read_text())
        flags = (summary['quick_mode'] and summary['correctness_only'] and summary['shared_gpu_mode'] and
                 not summary['timing_valid'] and not summary['accepted'])
        if args.step == 5:
            ratios = summary.get('deterministic_step_time_ratio') or {}
            if not (summary['all_checks_passed'] and flags and set(ratios) == set(summary['layouts_requested'])):
                raise RuntimeError('incomplete Step 5 deterministic-contract checks')
            for layout, ratio in ratios.items():
                value = ratio.get('median_ratio')
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise RuntimeError(f'missing or invalid deterministic step-time ratio for {layout}')
            status['layouts'] = summary['layouts_requested']
            status['deterministic_step_time_ratio'] = {layout: ratio['median_ratio'] for layout, ratio in ratios.items()}
        else:
            if not (summary['all_cases_passed'] and flags and summary['emulate'] == summary['real'] == 'passed'):
                raise RuntimeError('incomplete emulate/real correctness matrix')
            if args.node == 'SL3060':  # The controller node owns the real-mode EDP1 pause metric.
                pause = summary['edp1_rejoin_pause_s']
                if not isinstance(pause, (int, float)) or not math.isfinite(pause) or pause < 0:
                    raise RuntimeError('missing or invalid edp1_rejoin_pause_s')
                status['edp1_rejoin_pause_s'] = pause
        status['result'] = 'PASS'
    except BaseException:
        status.update(result='FAIL', error=traceback.format_exc())
    finally:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=30)  # run_case performs registered-PID cleanup.
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
                status.update(result='FAIL', cleanup_error='runner cleanup exceeded deadline; inspect registry')
        if restore:
            try:
                git('checkout', '--detach', BASE)
                if git('rev-parse', 'HEAD') != BASE:
                    raise RuntimeError('restore identity mismatch')
                status['restored_commit'] = BASE
            except BaseException:
                status.update(result='FAIL', restore_error=traceback.format_exc())
        status['elapsed_s'] = time.monotonic() - started
        if 'peer_wait_start' in status and 'peer_wait_s' not in status:
            status['peer_wait_s'] = time.monotonic() - status['peer_wait_start']
        if status['elapsed_s'] > 300:
            status.update(result='FAIL', budget_error='quick launch exceeded 300 seconds')
        try:
            record = run / args.node if (run / args.node).exists() else args.log_dir
            (record / 'quick_status.json').write_text(json.dumps({**status, 'checks_result': status['result'],
                'result': 'PENDING_FINALIZATION', 'receipt_required': True}, indent=2) + '\n')
            archive = run.parent / f'{run.name}_{args.node}.tar.gz'
            with tarfile.open(archive, 'x:gz') as output:
                output.add(record, arcname=f'{run.name}/{args.node}')
            status['archive'] = str(archive)
        except BaseException:
            status.update(result='FAIL', package_error=traceback.format_exc())
        if stream is not None:
            try:
                conn.settimeout(max(1, deadline - time.monotonic()))
                send(stream, status)
                peer = receive(stream)
                if peer['result'] != 'PASS':
                    status.update(result='FAIL', peer_error=peer)
            except BaseException:
                status.update(result='FAIL', terminal_error=traceback.format_exc())
            stream.close()
            conn.close()
        if time.monotonic() > deadline:
            status.update(result='FAIL', budget_error='quick finalization exceeded 300 seconds')
        receipt = run.parent / f'{run.name}_{args.node}.receipt.json'
        status['receipt'] = str(receipt)
        receipt.write_text(json.dumps(status, indent=2) + '\n')
        (args.log_dir / 'final.json').write_text(json.dumps(status, indent=2) + '\n')
        code = 0 if status['result'] == 'PASS' else 1
        (args.log_dir / 'exit_code').write_text(str(code) + '\n')
        print(json.dumps(status, indent=2), flush=True)
        if args.step == 5:
            print(f"QUICK CHECK step=5 result={status['result']} (memory preflight, deterministic reference/candidate "
                  f"{status.get('layouts')}, step_time_ratio={status.get('deterministic_step_time_ratio')}) "
                  f"shared_gpu_mode=true fail_stop_integration=pending logs={run}", flush=True)
        else:
            print(f"QUICK CHECK step={args.step} result={status['result']} (emulate shrink/expand, real shrink/expand) shared_gpu_mode=true edp1_rejoin_pause_s={status.get('edp1_rejoin_pause_s')} logs={run}", flush=True)
        lock.close()
    return code


if __name__ == '__main__':
    def stop(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    sys.exit(main())
