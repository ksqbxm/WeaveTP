import os, sys, numpy as np
def cpus(n):
    out = []
    for part in open(f"/sys/devices/system/node/node{n}/cpulist").read().strip().split(","):
        a, _, b = part.partition("-"); out += range(int(a), int(b or a) + 1)
    return out
mode, home, idx = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
src_node = 1 - home if mode == "upi" else home
N = 128 << 20
os.sched_setaffinity(0, {cpus(src_node)[-1 - idx]})
src = np.ones(N, dtype=np.uint8)
os.sched_setaffinity(0, {cpus(home)[-1 - idx]})
dst = np.ones(N, dtype=np.uint8)
print("ready", flush=True)
while True:
    np.copyto(dst, src)
