"""CPU-only per-node owner; kept outside the worker kill set."""

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from common import append_json, connect, host_backing, write_json
from safety import parent_death_signal
from source_fixture import create_checkpoint, demand

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from megatron.core.resharding.host_weights import HostOwner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    parent_death_signal()
    config = json.loads(Path(args.config).read_text())
    root = Path(config['output'])
    _, bus = connect(config)
    manifest = create_checkpoint(root / 'checkpoint', config['node'], config['source_width'])
    selected = demand(json.loads(manifest.read_text())['entries'])
    backing = host_backing(config)
    # Every local worker proves file-backed registration before any weight is read.
    bus.wait([f"host_registration/0/{r['slot']}" for r in config['topology'] if r['node'] == config['node']],
             timeout=config['startup_timeout_s'])
    owner = HostOwner(manifest, backing, config['host_library_mode'], selected,
                      record_created=lambda r: write_json(root / 'host_backing_identity.json', r))
    write_json(root / 'host_library.json', owner.report)
    bus.put(f"host_ready/{config['node']}", owner.report)
    try:
        deadline = time.monotonic() + config['launch_timeout_s']
        while not bus.has(f"host_stop/{config['node']}"):
            if time.monotonic() >= deadline:
                raise TimeoutError('host owner lifetime exceeded launch budget')
            time.sleep(0.05)
        owner.verify_unchanged()
        write_json(root / 'host_integrity.json', {'unchanged': True, 'owner_pid': owner.report['owner_pid'],
                                                'checkpoint_hash': owner.report['checkpoint_hash']})
    finally:
        owner.close()
        append_json(root / 'host_lifecycle.jsonl', {'action': 'owner_unlink', 'backing': str(backing),
                    'owner_pid': owner.report['owner_pid'], 't_local': time.monotonic()})


if __name__ == '__main__':
    def stop(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    main()
