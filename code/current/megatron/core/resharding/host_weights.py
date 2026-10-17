"""One CPU owner per node, bounded raw-checkpoint reads into one shared mmap.

The manifest is an explicit shard index (dtype, shape, expert and global range).
Each shard file is contiguous bytes with its own SHA256. On-demand selection is
resolved before any weight file is opened. There is no torch.load/full state_dict
temporary and no worker-private copy or worker-owned unlink.
"""

import hashlib
import json
import mmap
import os
import shutil
from pathlib import Path

import torch


def budget(manifest, mode, selected, *, ram_free, shm_free, memlock):
    if mode not in ('full', 'on-demand'):
        raise ValueError('host library mode must be full or on-demand')
    entries = manifest['entries']
    ids = [e['id'] for e in entries]
    if len(ids) != len(set(ids)) or not set(selected) <= set(ids):
        raise ValueError('invalid on-demand coverage list')
    chosen = entries if mode == 'full' else [e for e in entries if e['id'] in selected]
    if not chosen:
        raise ValueError('empty host coverage')
    size = sum(e['nbytes'] for e in chosen)
    report = {'mode': mode, 'full_backing_bytes': sum(e['nbytes'] for e in entries),
              'backing_bytes': size, 'worker_registration_bytes': size,
              'ram_available_bytes': ram_free, 'shm_available_bytes': shm_free,
              'memlock_bytes': memlock, 'coverage': [e['id'] for e in chosen]}
    # readinto fills the tmpfs backing directly, so the shared file is the whole
    # peak. cudaHostRegister pins through the driver, not RLIMIT_MEMLOCK: the
    # limit is recorded, and registration is proven by probe_registration.
    if ram_free < size or shm_free < size:
        raise MemoryError(f'host library budget rejected before reading weights: {report}')
    return chosen, report


def linux_capacity(directory):
    import resource

    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    return {'ram_free': available, 'shm_free': shutil.disk_usage(directory).free,
            'memlock': resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]}


def probe_registration(path):
    """Register one page of a shared file beside the backing: the mapping kind workers pin."""
    path = Path(path)
    with path.open('x+b') as file:
        try:
            file.truncate(mmap.PAGESIZE)
            mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE)
            try:
                pointer = torch.frombuffer(mapping, dtype=torch.uint8).data_ptr()
                result = torch.cuda.cudart().cudaHostRegister(pointer, mmap.PAGESIZE, 0)
                if result != 0:
                    raise RuntimeError(f'file-backed host registration preflight failed: {result}')
                result = torch.cuda.cudart().cudaHostUnregister(pointer)
                if result != 0:
                    raise RuntimeError(f'host unregister preflight failed: {result}')
            finally:
                mapping.close()
        finally:
            path.unlink()
    return {'registered_bytes': mmap.PAGESIZE, 'probe_path': str(path), 'backing_kind': 'shared_file'}


class HostOwner:
    def __init__(self, manifest_path, backing, mode, selected, *, capacity=None, record_created=lambda _: None):
        self.path = Path(backing)
        self.file = self.mapping = None
        manifest_path = Path(manifest_path)
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
        chosen, self.report = budget(manifest, mode, selected, **(capacity or linux_capacity(self.path.parent)))
        self.report.update(owner_pid=os.getpid(), node=manifest['node'], backing=str(self.path),
                           manifest_path=str(manifest_path), checkpoint_hash=hashlib.sha256(raw).hexdigest(),
                           structure=manifest['entries'], entries=[],
                           readonly_contract='source_only_views_with_owner_hash_verification')
        # Exclusive creation prevents accidental reuse/unlink of another launch.
        self.file = self.path.open('x+b')
        try:
            record_created({'backing': str(self.path), 'backing_inode': os.fstat(self.file.fileno()).st_ino,
                            'owner_pid': os.getpid()})
            self.file.truncate(self.report['backing_bytes'])
            self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_WRITE)
            offset = 0
            for entry in chosen:
                nbytes = entry['nbytes']
                with (manifest_path.parent / entry['file']).open('rb', buffering=0) as source:
                    if os.fstat(source.fileno()).st_size != nbytes:
                        raise ValueError('checkpoint shard size mismatch')
                    with memoryview(self.mapping)[offset:offset + nbytes] as target:
                        cursor = 0
                        while cursor < nbytes:
                            with target[cursor:cursor + (1 << 20)] as chunk:
                                count = source.readinto(chunk)
                            if not count:
                                raise EOFError('short checkpoint read')
                            cursor += count
                        if hashlib.sha256(target).hexdigest() != entry['sha256']:
                            raise ValueError('checkpoint shard hash mismatch')
                self.report['entries'].append({**entry, 'offset': offset})
                offset += nbytes
            self.mapping.flush()
            stat = os.fstat(self.file.fileno())
            self.report.update(actual_backing_bytes=stat.st_size, backing_inode=stat.st_ino)
            if Path('/proc/self/status').exists():
                self.report['owner_memory'] = [line for line in Path('/proc/self/status').read_text().splitlines()
                                               if line.startswith(('VmRSS:', 'VmHWM:', 'RssShmem:'))]
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        if self.file is not None:
            self.file.close()
            self.file = None
            self.path.unlink()

    def verify_unchanged(self):
        for entry in self.report['entries']:
            with memoryview(self.mapping)[entry['offset']:entry['offset'] + entry['nbytes']] as view:
                if hashlib.sha256(view).hexdigest() != entry['sha256']:
                    raise RuntimeError('read-only host weight contract violated')


class HostAttachment:
    """Source-only shared views; cudaHostRegister per process; owner alone unlinks.

    Registration uses a writable MAP_SHARED mapping, not MAP_PRIVATE/COW. CUDA
    pinning of OS-read-only mappings requires an optional device capability.
    No executor writes these views; the owner verifies every shard hash at exit.
    """

    def __init__(self, report, *, register=True, record=lambda **_: None):
        self.report, self.record = report, record
        self.file = Path(report['backing']).open('r+b')
        stat = os.fstat(self.file.fileno())
        if stat.st_ino != report['backing_inode'] or stat.st_size != report['backing_bytes']:
            self.file.close()
            raise RuntimeError('host backing identity/size changed')
        self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_WRITE)
        self.tensors = {}
        self.registered = False
        self.record(action='attach', pid=os.getpid(), owner_pid=report['owner_pid'],
                    backing=report['backing'], backing_inode=stat.st_ino)
        try:
            for entry in report['entries']:
                self.tensors[entry['id']] = torch.frombuffer(
                    self.mapping, dtype=getattr(torch, entry['dtype']), offset=entry['offset'],
                    count=entry['nbytes'] // entry['element_size']).view(entry['local_shape'])
            self.pointer = torch.frombuffer(self.mapping, dtype=torch.uint8).data_ptr()
            if register:
                result = torch.cuda.cudart().cudaHostRegister(self.pointer, stat.st_size, 0)
                if result != 0:
                    raise RuntimeError(f'cudaHostRegister failed: {result}; bytes={stat.st_size}')
                self.registered = True
                if not all(tensor.is_pinned() for tensor in self.tensors.values()):
                    raise RuntimeError('registered host views are not recognized as pinned')
                self.record(action='register', pid=os.getpid(), bytes=stat.st_size,
                            backing=report['backing'], owner_pid=report['owner_pid'])
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.registered:
            torch.cuda.synchronize()  # Last H2D use must finish before unregister.
            result = torch.cuda.cudart().cudaHostUnregister(self.pointer)
            if result != 0:
                raise RuntimeError(f'cudaHostUnregister failed: {result}')
            self.registered = False
            self.record(action='unregister', pid=os.getpid(), bytes=self.report['backing_bytes'])
        self.tensors.clear()
        self.mapping.close()
        self.file.close()
        self.record(action='detach', pid=os.getpid(), backing=self.report['backing'])
