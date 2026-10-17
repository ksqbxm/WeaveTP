"""Five-minute opt-in correctness launcher, fixed commit, peer handshake and restore."""

import argparse
import json
import re
import signal
import socket
import subprocess
import sys
import tarfile
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


def rendezvous(node, sha, port=29839):
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
        if request != {'node': 'SL3061', 'sha': sha, 'step': 2}:
            raise RuntimeError('peer step/SHA/node mismatch')
        suite = 's2q_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
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
        send(stream, {'node': node, 'sha': sha, 'step': 2})
        reply = receive(stream)
        if reply['sha'] != sha or not re.fullmatch(r's2q_[A-Za-z0-9_]+', reply['suite']):
            raise RuntimeError('invalid peer run ID/SHA')
        suite = reply['suite']
    return conn, stream, suite


def main():
    import fcntl

    parser = argparse.ArgumentParser()
    parser.add_argument('--node', choices=('SL3060', 'SL3061'), required=True)
    parser.add_argument('--sha', required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--log-dir', type=Path, required=True)
    args = parser.parse_args()
    started, deadline = time.monotonic(), time.monotonic() + 300
    run = args.log_dir
    conn = stream = child = None
    restore = False
    status = {'step': 2, 'node': args.node, 'quick_mode': True, 'shared_gpu_mode': True, 'result': 'FAIL'}
    lock = (args.log_dir.parent / f'{args.node}.lock').open('w')

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
        conn, stream, suite = rendezvous(args.node, args.sha)
        status['peer_wait_s'] = time.monotonic() - status['peer_wait_start']
        run = args.log_dir.parent.parent / suite
        status['suite'], status['run'] = suite, str(run)
        (args.log_dir / 'run_path.txt').write_text(str(run) + '\n')
        entry = args.repo / 'code/current/tools/resharding/failstop/run_topology.py'
        argv = [sys.executable, '-B', str(entry), '--node', args.node, '--suite-id', suite,
                '--output', str(run), '--mode', 'both', '--rounds', '1', '--real-launches', '1',
                '--quick', '--allow-foreign-gpu-processes', '--budget-seconds', str(max(1, deadline - time.monotonic() - 45))]
        child = subprocess.Popen(argv)
        code = child.wait(timeout=max(1, deadline - time.monotonic() - 40))
        if code:
            progress = run / args.node / 'progress.json'
            if progress.exists():
                status['failed_cases'] = [r for r in json.loads(progress.read_text()) if r['status'] != 'passed']
            raise RuntimeError(f'topology suite exited {code}; inspect per-round worker_error/controller_result logs')
        summary = json.loads((run / args.node / 'summary.json').read_text())
        if not (summary['all_cases_passed'] and summary['quick_mode'] and
                summary['emulate'] == summary['real'] == 'passed'):
            raise RuntimeError('incomplete emulate/real correctness matrix')
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
        print(f"QUICK CHECK step=2 result={status['result']} (emulate shrink/expand, real shrink/expand) shared_gpu_mode=true logs={run}", flush=True)
        lock.close()
    return code


if __name__ == '__main__':
    def stop(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    sys.exit(main())
