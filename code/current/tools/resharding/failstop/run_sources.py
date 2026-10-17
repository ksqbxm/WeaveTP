"""Step 3 source/transport suite, sharing the Step 2 fault protocol."""

import signal
import sys

from run_topology import main

if __name__ == '__main__':
    def stop(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    sys.exit(main(step=3))
