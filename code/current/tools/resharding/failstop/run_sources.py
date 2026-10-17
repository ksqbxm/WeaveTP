"""Step 3 source/transport suite; --profile-only collects the world16 profile without faults."""

import sys

from run_topology import main

if __name__ == '__main__':
    sys.exit(main(step=3))
