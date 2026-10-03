"""Explicit manual isolation only; run_t09 never calls this command."""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import t09


def case_processes(out, proc=Path('/proc')):
    found = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = [s.decode() for s in (entry / 'cmdline').read_bytes().split(b'\0') if s]
        except (FileNotFoundError, ProcessLookupError):
            continue
        if out in args or out + '/result.json' in args:
            found.append(int(entry.name))
    return found


def node_action(payload, rank, action):
    request = payload['request']
    root, source, target = (Path(payload[k]) for k in ('root_out', 'source', 'target'))
    lock = root / 'isolate.lock'
    controller_lock = root / 'compare.lock'
    t09.formal_path(str(root), payload['work'])
    t09.require(socket.gethostname().split('.')[0].lower() == request['nodes'][rank]['hostname'].lower(), 'wrong host')
    t09.require(source == root / request['case'] / f"r{request['repeat']}"
                and target.parent == root / '_failed' and target.name == payload['transaction'], 'invalid isolation paths')
    t09.require(source.resolve() == source and target.resolve() == target, 'isolation paths must not use symlinks')
    if action != 'prepare':
        t09.require(t09.read(controller_lock) == payload, 'controller/isolation lock mismatch')
    t09.require(not case_processes(str(source)), 'case still has tagged processes; isolation does not kill them')
    if action == 'prepare':
        t09.require(source.is_dir() and not target.exists(), 'source missing or target exists')
        t09.require(not (source / 'complete.json').exists(), 'refuse to isolate a completed case')
        saved = t09.read(source / 'request.json')
        t09.require(saved['case'] == request['case'] and saved['repeat'] == request['repeat']
                    and saved['out_dir'] == str(source), 'case request identity mismatch')
        # Same exclusive lock as the controller closes the check/start race.
        t09.compare.save(controller_lock, payload)
        t09.compare.save(lock, payload)
        target.parent.mkdir(exist_ok=True)
        t09.require(source.stat().st_dev == target.parent.stat().st_dev, 'rename must stay on same filesystem')
    else:
        t09.require(t09.read(lock) == payload, 'isolation transaction lock mismatch')
        if action == 'move':
            t09.require(source.is_dir() and not target.exists(), 'source/target changed after prepare')
            t09.compare.save(source / f"isolation.{payload['transaction']}.json",
                             {'reason': payload['reason'], 'source': str(source), 'target': str(target),
                              'rank': rank, 'transaction': payload['transaction']})
            source.rename(target)
        elif action == 'verify':
            t09.require(target.is_dir() and not source.exists(), 'move unconfirmed')
        elif action == 'rollback':
            if target.exists():
                t09.require(not source.exists(), 'both source and target exist; preserve evidence')
                target.rename(source)
            t09.require(source.is_dir() and not target.exists(), 'rollback unconfirmed')
        elif action == 'unlock':
            lock.unlink()
            controller_lock.unlink()
        else:
            raise ValueError('unknown isolation action')
    return {'rank': rank, 'action': action, 'ok': True}


def rpc(payload, rank, action):
    command = t09.node_command(payload, rank, ['isolate_failed.py', '--node-rank', str(rank), '--action', action])
    result = subprocess.run(command, input=json.dumps(payload), capture_output=True, text=True,
                            encoding='utf-8', timeout=90)
    t09.require(result.returncode == 0, f'node{rank} {action} failed: {result.stderr}')
    receipt = json.loads(result.stdout)
    t09.require(receipt == {'rank': rank, 'action': action, 'ok': True}, 'unexpected isolation receipt')


def coordinate(payload, call=rpc):
    prepared = []
    try:
        for rank in (0, 1):
            # Include an ambiguous prepare response in compensation; never assume no mutation.
            prepared.append(rank)
            call(payload, rank, 'prepare')
        for rank in (0, 1):
            call(payload, rank, 'move')
        for rank in (0, 1):
            call(payload, rank, 'verify')
    except BaseException as original:
        errors = []
        for rank in prepared:
            try:
                call(payload, rank, 'rollback')
            except BaseException as exc:
                errors.append(f'node{rank}: {exc}')
        if errors:
            raise RuntimeError('isolation/rollback UNCONFIRMED; preserve locks; no resume: ' + '; '.join(errors)) from original
        for rank in prepared:
            call(payload, rank, 'unlock')
        raise RuntimeError(f'isolation failed; both original locations restored: {original}') from original
    for rank in (0, 1):
        call(payload, rank, 'unlock')
    print(f"ISOLATED both nodes: {payload['target']} reason={payload['reason']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', nargs='?')
    parser.add_argument('--node-rank', type=int, choices=(0, 1))
    parser.add_argument('--action', choices=('prepare', 'move', 'verify', 'rollback', 'unlock'))
    args = parser.parse_args()
    if args.node_rank is not None:
        print(json.dumps(node_action(json.load(sys.stdin), args.node_rank, args.action)))
        return
    t09.require(socket.gethostname().split('.')[0].lower() == 'sl3060', 'manual command requires SL3060')
    t09.require(sys.stdin.isatty(), 'manual isolation requires an interactive terminal and typed reason')
    settings = t09.compare.settings_from_env()
    t09.formal_path(settings['root_out'], os.environ['WORK'])
    t09.require(args.case, 'specify <case>_r<k>')
    request = t09.selected_cases(settings, args.case)[0]
    reason = input(f"Isolation reason for {args.case} (empty cancels): ").strip()
    t09.require(reason, 'cancelled: reason required')
    root = Path(settings['root_out'])
    transaction = args.case + '_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    coordinate({'request': request, 'root_out': str(root), 'work': os.environ['WORK'],
                'source': request['out_dir'], 'target': str(root / '_failed' / transaction),
                'transaction': transaction, 'reason': reason})


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
