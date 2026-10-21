"""Keep survivor rank order across real rejoin so TP2 can continue serving."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from megatron.core.failstop import FaultSession  # noqa: E402


class AsyncSession(FaultSession):
    def rejoin(self, ranks, event=None):
        super().rejoin(ranks, event)
        if self.backend.mode == "real":
            # Returned empty receivers follow the twelve surviving ranks.
            # PyTorch sorts group ranks; this preserves live TP2 coordinates.
            self.members = self.previous + tuple(s for s in range(16) if s not in self.previous)
        return self.current_world()
