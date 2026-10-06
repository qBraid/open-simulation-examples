---
name: solver-racing
description: Race several optimization solvers on the same problem under one time budget (as local processes, or one per qBraid instance), keep the best answer, and certify it with a lower bound. Use when a routing, MIP, LP or scheduling answer must be good and on time, when no single open-source solver is reliably best, or when a user asks how open-source can compete with Gurobi.
metadata:
  version: "0.3.0"
  layer: "method"
  status: "draft"
  verified: "2026-10-01"
---

# Solver racing

No open-source solver wins every instance. Racing turns that into an advantage:
run the candidates at once, keep the best incumbent, and let the exact solver
prove how far from optimal it can be.

Start from the user's goal: if it is unclear whether they want the best solution
or the best quantum solution, ask (see solution-router).

## Get the scripts

The reference implementation is `fleet-routing/race.py` (plus `bench.py` for
checkpointed benchmark runs) in https://github.com/qBraid/open-simulation-examples:

```bash
git clone --depth 1 https://github.com/qBraid/open-simulation-examples
cd open-simulation-examples/fleet-routing
```

The environment is the one in **optimization-stack**: check `qbraid envs available`
first, otherwise build it from `requirements.txt` (and `requirements-gpu.txt` for
cuOpt), and package it with the **manage-environments** skill for reuse.

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

```bash
python race.py data/A-n32-k5.vrp --time 60 --bks 784 --out results/a-n32-k5.json
```

The same pattern applies beyond routing: energy-systems races AC-OPF
formulations and HiGHS simplex against IPM.

## Mapping to qBraid compute

- Local processes on one machine are the default. The subscription pod handles
  3 racers x 1 thread.
- One racer per instance: at most 5 instances can run at once, and the
  subscription pod counts. Run `qbraid compute status` before launching; if the
  account is at its limit, race locally.
- Put the CPU racers on `cpu-8v-32g` or `cpu-32v-128g`, and cuOpt on `gpu-l4`
  ($0.49/h) or `gpu-h100-sxm` ($5.37/h). Launch them through
  **qbraid-cloud-orchestration**, set `qbraid compute instances set <label> --auto-stop <min>`,
  and terminate when done.
- A `gpu-l4` instance's cgroup allows about 5 CPUs even though `nproc` shows 48.
  Size CPU racers on it to the cgroup, not to `nproc`.
- Share incumbents through files that come back to the coordinator; do not stream
  solutions between instances mid-race unless the solver accepts warm starts.

## Does racing pay? (measured 2026-10-01)

| Budget (30 CVRPLIB X instances) | PyVRP alone | Race | Who wins |
|---|---|---|---|
| 10% of the published 2.4·n s | 0.60% | **0.49%** (+ cuOpt on one L4) | cuOpt wins 10 of 30, mostly n < 500; OR-Tools never |
| Full published budget | 0.34% | **0.31%** | cuOpt (at 10% time) still wins 5 of 30 |

GPU and CPU racers don't compete for cores, so adding cuOpt costs nothing extra
in wall time. Gaps are the mean gap to best-known; details in
`fleet-routing/README.md`.

## Long runs

- Checkpoint one JSON per instance and skip finished ones on restart (`bench.py`).
  A restart killed a run halfway; it resumed on another machine with nothing lost.
- Keep heavy jobs off the subscription pod and in chunks of 60 minutes or less.
  An out-of-memory kill on the pod takes down every agent running there.
- `pkill -f <pattern>` and `pgrep -f` over `ssh host '...'` match the ssh command
  line itself and kill your own session. Use bracketed patterns (`[b]ench.py`) or PIDs.

## Traps (measured)

- `highspy` and `ortools` in one process: undefined-symbol import errors. Keep
  them in separate processes.
- Racers slow each other down on a shared machine. HiGHS proved A-n32-k5 in 13 s
  alone and in about 30 s with two other racers alongside.
- Ties: credit the solver that reached the best cost first, not dict order.
- Asymmetric (road) distances: an undirected MIP on `min(d_ij, d_ji)` is a
  valid *relaxation* (the bound holds), but re-cost its routes directed and
  orient each route in its cheaper direction. A directed bound is much tighter
  (5.15% vs 8.4% certified gap on the Chicago example).

## Quantum: the short answer

**Runnable test today.** A QPU or simulator sampler can join the race as one more
racer, scored by the same certified gap; today it will not win (see the QAOA
readiness result in optimization-stack), and the race makes that visible honestly.

For the full answer and the experiment protocol, use the **quantum-readiness**
skill (section "Optimization").

## Verification stamp

- 2026-10-01: racing PyVRP 0.14, OR-Tools 9.15 and cuOpt 26.8 (on a `gpu-l4` instance) on 30 CVRPLIB X instances at 0.1 x 2.4·n s gives a 0.49% mean gap; at the full budget (pinned cores on a `cpu-32v-128g` instance, 54 min wall) 0.31%.
- 2026-10-01: the directed Chicago bound certifies 5.15% (900 s, 2 threads).
- 2026-09-30, subscription pod, 0 credits: on X-n101-k25 PyVRP reached the best-known 27591 at 30 s, OR-Tools was at 29159 (+5.7%) at 60 s, and HiGHS certified no solution below 26034 (5.6% gap) after a 240 s certify step.
