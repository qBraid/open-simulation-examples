"""Full-budget CVRPLIB X run: PyVRP, one instance per core, longest first.

Each instance gets the published Tmax = 2.4 n s on one pinned core (taskset), so
instances never share a core. Results are checkpointed per instance by bench.py.

    python fullbench.py --cores 20-29 --list data/X/xsub.txt --dir data/X --out results/xbench_full30
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--factor", type=float, default=1.0)
    ap.add_argument("--host", default="dedicated qBraid cpu box (32 vCPU), 1 pinned core per instance")
    a = ap.parse_args()
    lo, hi = map(int, a.cores.split("-"))
    cores = Queue()
    for c in range(lo, hi + 1):
        cores.put(c)
    names = [l.strip() for l in open(a.list) if l.strip()]
    names.sort(key=lambda s: -int(re.search(r"n(\d+)", s).group(1)))  # longest first
    tmp = os.path.join(a.out, "_lists")
    os.makedirs(tmp, exist_ok=True)

    def run(name):
        if os.path.exists(os.path.join(a.out, f"{name}.json")):
            return name, 0
        c = cores.get()
        try:
            lst = os.path.join(tmp, name + ".txt")
            open(lst, "w").write(name + "\n")
            cmd = ["taskset", "-c", str(c), sys.executable, "bench.py", "--list", lst, "--dir", a.dir,
                   "--factor", str(a.factor), "--solvers", "pyvrp", "--sequential", "--out", a.out,
                   "--host", f"{a.host} (core {c})"]
            r = subprocess.run(cmd, env={**os.environ, "OMP_NUM_THREADS": "1"}, capture_output=True, text=True)
            print(r.stdout.strip() or r.stderr[-300:], flush=True)
            return name, r.returncode
        finally:
            cores.put(c)

    with ThreadPoolExecutor(hi - lo + 1) as ex:
        for name, rc in ex.map(run, names):
            pass
    print("FULL30DONE", flush=True)


if __name__ == "__main__":
    main()
