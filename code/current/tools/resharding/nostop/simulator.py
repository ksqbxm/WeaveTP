"""Read-only loading of the existing CPU simulator and its input locations."""

import importlib.util
import sys
from pathlib import Path

SIMULATOR = Path(__file__).resolve().parents[5] / "documents/e2e_prep_20261008/sim/e2e_sim.py"
PROFILE_DATA = Path("/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile")


def load_simulator(path=SIMULATOR):
    spec = importlib.util.spec_from_file_location("nostop_e2e_sim", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolves the defining module.
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True  # Analysis must not alter the simulator inputs.
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    module.set_world(8)
    return module
