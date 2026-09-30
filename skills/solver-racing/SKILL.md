---
name: solver-racing
description: Race several optimization solvers on the same problem under one time budget (as local processes, or one per qBraid instance), keep the best answer, and certify it with a lower bound. Use when a routing/MIP/scheduling answer must be good and on time, when no single open-source solver is reliably best, or when a user asks how open-source can compete with Gurobi.
metadata:
  version: "0.1.0"
  layer: "2"
  status: "draft"
  verified: "2026-09-30"
---

# Solver racing

No open-source solver wins every instance. Racing turns that into an advantage:
run the candidates at once, keep the best incumbent, and let the exact solver
prove how far from optimal it can be.

## Shape

1. **Racers** in separate processes, each with the same wall-clock budget and
   1 thread (or pinned cores). Heuristics (PyVRP, OR-Tools, cuOpt) give
   incumbents; the exact MIP (HiGHS) gives a lower bound.
2. **Log every improvement** as `[t, incumbent, lower_bound]`, so the
   convergence plot and "gap at 1/5/10/30/60 s" tables come for free.
3. **Certify.** After the race, hand the winning solution to HiGHS as an upper
   bound and MIP start for a fixed extra budget (`--certify`). Report
   `best`, `lower_bound`, `certified_gap = (best - lb) / best`. With integer
   costs, `ceil(lb) >= best` is a proof of optimality.
4. **Check** the winner independently (every customer once, capacity, recomputed
   cost) before reporting it.

Reference implementation: `fleet-routing/race.py` in
`qBraid/open-simulation-examples`.

## Mapping to qBraid compute

- Local processes on one box are the default. The pod handles 3 racers × 1 thread.
- One racer per instance: at most 5 instances can run at once, and the
  subscription pod counts. Run `qbraid compute status` before launching; if the
  account is at its limit, race locally.
  Put the CPU racers on `cpu-8v-32g` / `cpu-32v-128g`, and cuOpt on `gpu-l4` ($0.49/h) or
  `gpu-h100-sxm` ($5.37/h). Set `qbraid compute instances set <label> --auto-stop <min>`
  and terminate when done.
- Share incumbents through files that come back to the coordinator; do not stream
  solutions between instances mid-race unless the solver accepts warm starts.

## Traps (measured)

- `highspy` and `ortools` in one process: undefined-symbol import errors. Keep
  them in separate processes.
- Racers slow each other down on a shared box. HiGHS proved A-n32-k5 in 13 s
  alone and in about 30 s with two other racers alongside.
- Ties: credit the solver that reached the best cost first, not dict order.
- Asymmetric (road) distances: an undirected MIP on `min(d_ij, d_ji)` is a
  valid *relaxation* (the bound holds), but re-cost its routes directed and
  orient each route in its cheaper direction.

## Verified

2026-09-30, qBraid subscription pod, 0 credits. On X-n101-k25: PyVRP reached the
best-known 27591 at 30 s and OR-Tools was at 29159 (+5.7%) at 60 s. HiGHS
certified that no solution is below 26034 (5.6% gap) after a 240 s certify step.
