"""Fail-slow co-runner (Malleus-style, no root needed): a bf16 matmul loop on the GPU in CUDA_VISIBLE_DEVICES.

  failslow_hog.py --tag <launch> --ready <file>

Level W/M/S runs 1/2/3 of these per slow GPU (failslow_inject). It touches its ready file after the first matmul,
stays under 1 GiB of device memory, dies with its supervisor (PR_SET_PDEATHSIG) and stops on SIGTERM. The
injector finds strays by this script's name plus --tag only.
"""

import argparse
import signal
from pathlib import Path

GIB = 1 << 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--ready', type=Path, required=True)
    parser.add_argument('--size', type=int, default=4096)
    args = parser.parse_args()
    from safety import parent_death_signal

    parent_death_signal()
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(True))
    import torch

    a = torch.randn(args.size, args.size, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(args.size, args.size, device='cuda', dtype=torch.bfloat16)
    c = torch.empty_like(a)
    torch.matmul(a, b, out=c)
    torch.cuda.synchronize()
    if torch.cuda.memory_reserved() > GIB:
        raise SystemExit(f'hog reserves {torch.cuda.memory_reserved()} bytes > 1 GiB')
    args.ready.write_text('ready\n')
    while not stop:
        for _ in range(16):
            torch.matmul(a, b, out=c)
        torch.cuda.synchronize()


if __name__ == '__main__':
    main()
