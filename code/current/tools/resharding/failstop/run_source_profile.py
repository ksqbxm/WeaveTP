"""Standalone profile collection, with the same isolation and owner preflight."""

import signal
import sys

from run_topology import main

if __name__ == '__main__':
    def stop(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    sys.exit(main(step=3, profile_only=True))
