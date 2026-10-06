"""Plug a sampler into QOBLIB: the shape a quantum or hybrid entry takes.

Replace `sample()` with your own sampler: a Qiskit SamplerV2 run on IBM hardware,
a QAOA simulation, or a qBraid device job. Everything else stays the same: the
QUBO comes from QOBLIB itself, the samples are scored and repaired the same way
for everyone, and the result goes through the official checker.

    python example_sampler.py 07-independentset MANN-a9 --shots 2000

The stand-in below draws uniform random bitstrings. It is a floor to beat, not a
quantum method: report any real sampler against both this floor and the
classical baseline (`python qoblib.py solve ...`).
"""
import argparse
import random
import time
from pathlib import Path

import qoblib


def sample(offset, q, n, shots, seed=7):
    """Stand-in sampler: replace with QPU or simulator output (lists of 0/1, variable 1 first)."""
    rng = random.Random(seed)
    return [[rng.randint(0, 1) for _ in range(n)] for _ in range(shots)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cls", choices=["01-marketsplit", "07-independentset"])
    ap.add_argument("instance")
    ap.add_argument("--shots", type=int, default=2000)
    args = ap.parse_args()

    inst = qoblib.load(args.cls, args.instance)
    offset, q = qoblib.load_qubo(args.cls, args.instance)
    n = max(j for _, j in q)
    t0 = time.time()
    samples = sample(offset, q, n, args.shots)
    energies = sorted(qoblib.qubo_energy(offset, q, s) for s in samples)
    sol, obj, feas = qoblib.best_of_samples(inst, samples)
    bkv = qoblib.best_known(args.cls).get(args.instance, ("?", "?"))
    print(f"{args.instance}: {n} QUBO variables, {args.shots} samples in {time.time() - t0:.1f} s")
    print(f"  lowest raw QUBO energy {energies[0]:g}; after post-processing objective {obj:g} "
          f"(feasible {feas}); best known {bkv[0]}")
    out = qoblib.write_solution(inst, sol, Path(f"{args.instance}.sampler.sol"))
    code, meaning = qoblib.run_official_check(inst, out)
    print(f"  official checker: {meaning}; solution -> {out}")


if __name__ == "__main__":
    main()
