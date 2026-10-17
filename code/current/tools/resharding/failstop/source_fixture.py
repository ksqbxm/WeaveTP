"""Indexed toy checkpoint and independent per-shard oracle for Step 3."""

import array
import hashlib
import json
from pathlib import Path

NAMES = ('gpu_only', 'host_only', 'mixed')


def identity(row):
    return f"{row['node']}/{row['gpu_uuid']}"


def host_resident(name, part):
    """Fixture placement: host_only lives only in the library; mixed alternates by part."""
    return name == 'host_only' or (name == 'mixed' and part % 2 == 0)


def create_checkpoint(root, node, width):
    root = Path(root)
    root.mkdir()
    entries = []
    for name_index, name in enumerate(NAMES):
        for expert in range(2):
            for part in range(8):
                start, stop = part * width // 8, (part + 1) * width // 8
                key = f'{name}_{expert}_{part}'
                path = root / f'{key}.bin'
                hash_ = hashlib.sha256()
                with path.open('xb') as stream:
                    for row in range(2):
                        for lo in range(start, stop, 4096):
                            values = array.array('q', (name_index * 10000000 + expert * 1000000 + row * width + col
                                                      for col in range(lo, min(stop, lo + 4096))))
                            raw = values.tobytes()
                            stream.write(raw)
                            hash_.update(raw)
                entries.append({'id': key, 'file': path.name, 'name': name, 'expert': expert,
                                'dtype': 'int64', 'element_size': 8, 'shape': [2, width],
                                'start': [0, start], 'stop': [2, stop], 'local_shape': [2, stop - start],
                                'nbytes': 2 * (stop - start) * 8, 'sha256': hash_.hexdigest(), 'part': part})
    manifest = {'version': 1, 'node': node, 'entries': entries}
    path = root / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2))
    return path


def demand(entries):
    return [e['id'] for e in entries if host_resident(e['name'], e['part'])]


def read_oracle(manifest_path, name, expert, start, stop):
    """Read checkpoint columns directly; no planner, source map or task reuse."""
    import torch

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    result = torch.empty((2, stop - start), dtype=torch.int64)
    covered = torch.zeros(stop - start, dtype=torch.bool)
    for entry in manifest['entries']:
        if entry['name'] != name or entry['expert'] != expert:
            continue
        lo, hi = max(start, entry['start'][1]), min(stop, entry['stop'][1])
        if lo >= hi:
            continue
        raw = (manifest_path.parent / entry['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('oracle checkpoint hash mismatch')
        data = torch.frombuffer(bytearray(raw), dtype=torch.int64).view(entry['local_shape'])
        result[:, lo - start:hi - start] = data[:, lo - entry['start'][1]:hi - entry['start'][1]]
        covered[lo - start:hi - start] = True
    if not covered.all():
        raise ValueError('oracle checkpoint coverage hole')
    return result
