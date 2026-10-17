"""Step 4 replica-local async driver and memory suite; the mechanism runner is async_worker.py."""

import sys

from run_topology import main

if __name__ == '__main__':
    sys.exit(main(step=4))
