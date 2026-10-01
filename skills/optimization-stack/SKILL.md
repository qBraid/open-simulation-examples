---
name: optimization-stack
description: Solve LP, MIP, vehicle-routing and scheduling problems on qBraid with open-source solvers (HiGHS, PyVRP, OR-Tools CP-SAT/routing, SCIP, NVIDIA cuOpt). Use when a user brings an optimization, routing, scheduling, supply-chain or planning problem, or asks for a Gurobi/CPLEX alternative. Gives the which-solver decision rules, the known-answer check to run first, verified costs, and where quantum (QUBO/QAOA) fits honestly.
metadata:
  version: "0.2.0"
  layer: "1"
  status: "draft"
  verified: "2026-10-01"
---

# Optimization stack on qBraid

Open-source solvers, chosen by problem class, checked against a known answer
before any claim. Pair with **solver-racing** when the answer matters and time
is bounded, and with **qbraid-cloud-orchestration** for instances.

## Pick the solver (decision rules)

| Problem | First choice | Also race | Notes |
|---|---|---|---|
| LP, any size | HiGHS | cuOpt (GPU PDLP) above ~1M nonzeros | On Mittelmann LPfeas (Jun 2026), cuOpt 26.06 ranked first. HiGHS solved 86% at 17x the leader's time. |
| MIP, general | HiGHS | SCIP | **This is a real gap.** On Mittelmann MIPLIB2017 (Apr 2026) HiGHS is 7.4x slower than the commercial leader and solves 68% vs 91%. Do not promise Gurobi-class MIP. |
| CVRP / VRPTW / routing | PyVRP | cuOpt (GPU), OR-Tools routing | On 30 CVRPLIB X instances at 10% of the published budget: PyVRP 0.60%, cuOpt (L4) 0.99%, OR-Tools 7.96% (no plan on 4 tight instances). Racing PyVRP + cuOpt: **0.49%**. |
| Scheduling, rostering, packing | OR-Tools CP-SAT | Timefold | CP-SAT is multi-threaded; give it the cores. |
| Convex (QP, SOCP) | CVXPY → Clarabel/HiGHS | cuOpt QP (beta) | |
| Modelling layer | Pyomo or CVXPY; plain highspy for cut loops | | |

Size thresholds, from verified runs: an exact two-index CVRP MIP with greedy
capacity cuts **proves optimality up to ~30 customers in seconds** (A-n32-k5
in 13 s alone). At 100 customers it leaves a ~5–6% certified gap. Closing that
needs branch-cut-and-price, which the open stack doesn't have in a ready-made form.

## Baseline first (mandatory)

Before any production or quantum claim, reproduce a known answer on the same
code path:

- CVRP: `E-n22-k4` (optimum 375) and `A-n32-k5` (optimum 784) must come out
  proven optimal; `X-n101-k25` must reach its best-known 27591.
- A heuristic result is reported with a **lower bound** from HiGHS, never as
  "optimal".

## Benchmark (CVRPLIB X, verified 2026-10-01)

The field's bar is the published gap to best-known solutions at Tmax = 2.4·n s:
HGS-CVRP 0.11%, PyVRP 0.22%. Measured here on 30 stratified instances
(n 100–1000), one seed, at **0.1 × that budget**:

| Size | Race (CPU + GPU) | PyVRP | cuOpt (L4) |
|---|---|---|---|
| n 100–299 (13) | 0.35% | 0.47% | 0.63% |
| n 300–599 (10) | 0.42% | 0.54% | 0.91% |
| n 600–1000 (7) | 0.87% | 0.92% | 1.78% |
| all 30 | **0.49%** | 0.60% | 0.99% |

**At the full published budget** (2.4·n s, one pinned core per instance, one seed;
dedicated 32-vCPU box, 54 min wall for all 30): PyVRP averages **0.34%**, and racing
the 10%-time cuOpt runs gives **0.31%**. By size:
- n < 300: 0.14%, beats the bar;
- n 300–599: 0.27%;
- n 600–1000: 0.69%.

Verdict: **close, not reached** (bar 0.22%). The gap is on the largest instances; one
seed and an uncalibrated CPU speed add noise. All 100 instances × 3 seeds costs
about $5 on `cpu-32v-128g`.

**Recipe:** `fullbench.py --cores 20-29` runs one instance per pinned core,
longest first.

**cuOpt trap.** cuOpt minimises **fleet size first**, then distance. On distance-only
benchmarks its default lost 13% on X-n101. Setting `min_vehicles = k_min + 1` reached
the best-known solution; an explicit zero fleet cost gave +0.2%. Choose the setting on
one instance, then freeze it. For real fleet-cost problems the default is the right one.

## Environment

`fleet-routing/requirements.txt` in `qBraid/open-simulation-examples` (Python 3.12,
CPU-only). cuOpt is GPU-only: `requirements-gpu.txt`
(`cuopt-cu12==26.8.0`, `--extra-index-url https://pypi.nvidia.com`), on
`gpu-l4` or `gpu-h100-sxm`. Not yet packaged as a qBraid env; when it is, install
it with `qbraid envs install <slug>`, not with pip into system Python.

**Trap:** for asymmetric (road) distances, bound the **directed** model. A symmetric
`min(d_ij, d_ji)` relaxation is valid but loose: 8.4% against 5.15% on Chicago.

**Trap:** `overpass-api.de` returns HTTP 406 to cloud IPs. Use the
`maps.mail.ru/osm/tools/overpass/api/interpreter` mirror, or a city open-data portal
for buildings.

**Trap:** `highspy` 1.15 and `ortools` 9.15 cannot be imported into the same
Python process. Each one's bundled HiGHS breaks the other's shared library
(undefined symbol at import). Run them in separate processes (the racing harness
does this).

## Verified recipe

Verified 2026-09-30 on the qBraid subscription pod: 3 processes × 1 thread,
60 s per race, 0 credits.

| Instance | Known | Best found | Lower bound | Status |
|---|---|---|---|---|
| E-n22-k4 | 375 | 375 | 375 | proven optimal |
| A-n32-k5 | 784 | 784 (PyVRP, <1 s) | 784 | proven optimal |
| X-n101-k25 | 27591 | 27591 (PyVRP, 30 s) | 26034 | certified gap 5.6% |
| Chicago, 80 stops, OSM roads | none | 145.7 km (PyVRP) | 138.2 km (directed ACVRP, `bound.py`, 900 s) | certified gap 5.15% (was 8.4% with a symmetric bound) |

```
python race.py data/A-n32-k5.vrp --time 60 --bks 784 --out results/a-n32-k5.json
python city.py --stops 80 --time 60
```

## Where quantum fits (say it this way)

- **Today:** QUBO/QAOA on sub-problems of about 16–20 variables, as a
  *readiness experiment* that is always scored against the brute-force or MIP
  optimum and by drawn shots. The example here runs QAOA on one 4-stop route
  (16 qubits). Measured: P(optimal) 0.11–0.14% at p = 1–3, which is 30–40x *below*
  choosing a random valid tour, because 97% of shots break the one-hot
  constraints. Prefer constraint-preserving mixers before buying QPU time.
  Formulation: one-hot stop × position with row and column
  penalties, and the depot fixed. See **qubo-formulation** when it exists.
- **Future:** Grover-type quadratic speedups inside tree search need fault
  tolerance, and overheads cancel them at most sizes. Quantum-inspired
  heuristics can be used today.
- Before any QPU run, price it with `qbraid devices get <qrn>` and confirm with
  the user. Example: IQM Garnet is 30 credits per task + 0.145 per shot, so
  1000 shots cost 175 credits ($1.75).
