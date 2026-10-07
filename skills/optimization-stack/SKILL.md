---
name: optimization-stack
description: Solve LP, MIP, vehicle-routing and scheduling problems on qBraid with open-source solvers (HiGHS, PyVRP, OR-Tools CP-SAT/routing, SCIP, NVIDIA cuOpt). Use when a user brings an optimization, routing, scheduling, supply-chain or planning problem (including turning it into a model), asks for a Gurobi/CPLEX alternative, or wants to know where QUBO/QAOA fits. Gives the which-solver decision rules, the known-answer check to run first, verified benchmark numbers and costs, the traps, and an honest, runnable quantum readiness test.
metadata:
  version: "0.4.0"
  status: "provisional"
  verified: "2026-10-01"
---

# Optimization stack on qBraid

Open-source solvers, chosen by problem class, checked against a known answer
before any claim. Pair it with **solver-racing** when the answer matters and time
is bounded, and with **qbraid-cloud-orchestration** for instances.

Start from the user's goal: if it is unclear whether they want the best solution
or the best quantum solution, ask (see solution-router).

## Get the scripts

The worked example lives in `fleet-routing/` of
https://github.com/qBraid/open-simulation-examples:

```bash
git clone --depth 1 https://github.com/qBraid/open-simulation-examples
cd open-simulation-examples/fleet-routing
```

All script references below are relative to that directory.

## Environment

Install the qBraid environment (kernel included):

- `qbraid envs install optimi_84bd83`: **optimization** (CPU): PyVRP 0.14.0, OR-Tools 9.15,
  HiGHS 1.15.1, cvxpy, Pyomo, OSMnx. 1.1 GB.
- `qbraid envs install optimi_2bgc8m`: **optimization-gpu**: the same plus NVIDIA cuOpt 26.8.0
  and cuDF (needs an NVIDIA GPU, CUDA 12, driver 535+). 7.2 GB. Import `pyomo` before `cuopt`.

Both verified from a fresh install on a `gpu-l4` instance on 2026-10-06 (E-n22-k4 and
A-n32-k5 proven optimal; cuOpt reached X-n101-k25's best-known 27591). Without qBraid,
build it from the example's pinned files:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # CPU: PyVRP, OR-Tools, HiGHS, Qiskit, osmnx
# GPU racer, separate venv, on a gpu-l4 or gpu-h100-sxm instance:
pip install --extra-index-url https://pypi.nvidia.com -r requirements-gpu.txt   # cuopt-cu12 26.8.0
```

To reuse it across instances, package it as a qBraid environment with the
**manage-environments** skill and install it with `qbraid envs install <slug>`,
rather than pip-installing into system Python.

## Solve the user's problem (start here)

Users bring a problem, not a model. Work through these four steps in order,
and show the user each one.

1. **Formulate.** Name the decisions (which truck serves which stop, who works
   which shift), the hard constraints, and the objective, in the user's units.
   Then classify it:
   - continuous and linear: LP;
   - yes/no or integer choices: MIP;
   - routes over a road network: CVRP/VRPTW;
   - shifts and sequences: CP-SAT scheduling;
   - pure pairwise yes/no interactions: QUBO.

   If the user trades off several objectives (cost vs service level vs carbon),
   say so. Report a Pareto front, not one weighted answer; this is also the
   multi-objective setting QAMOO targets. Start from a small instance the user
   can check by hand, then scale it up.
2. **Solve it classically, to a known quality.** Use the decision rules below.
   Report the best solution *and* a lower bound, so the user knows how far from
   optimal it can be. Race solvers when no single one is reliably best
   (solver-racing).
3. **Find the quantum-sized piece** (only if the user wants quantum, or both).
   Look for a sub-problem with about 12-50 binary variables that is still
   meaningful, such as one route, one shift block, or one asset subset. Write its QUBO and
   check the QUBO's optimum matches the sub-problem's. Prefer
   constraint-preserving mixers over large penalties (see the Quantum section).
4. **Compare honestly.** Score the quantum result against the classical optimum
   on the same instance, with the same metric. For public claims, use QOBLIB:
   its per-class checkers and leaderboard are the field's yardstick
   (`quantum-optimization/qoblib/` in the repo above).

## Pick the solver (decision rules)

| Problem | First choice | Also race | Notes |
|---|---|---|---|
| LP, any size | HiGHS | cuOpt (GPU PDLP) above ~1M nonzeros | cuOpt ranked first on the Mittelmann LPfeas benchmark in June 2026 ([plots](https://mattmilten.github.io/mittelmann-plots/)); check the current table before quoting a ratio. |
| MIP, general | HiGHS | SCIP | **This is a real gap.** On Mittelmann MIPLIB2017 (Apr 2026; [plots](https://mattmilten.github.io/mittelmann-plots/)) HiGHS is 7.4x slower than the commercial leader and solves 68% vs 91%. Do not promise Gurobi-class MIP. |
| CVRP / VRPTW / routing | PyVRP | cuOpt (GPU), OR-Tools routing | Racing PyVRP with cuOpt beats either alone (benchmark below). OR-Tools routing is several percent behind on CVRPLIB X. |
| Scheduling, rostering, packing | OR-Tools CP-SAT | Timefold | CP-SAT is multi-threaded; give it the cores. |
| Convex (QP, SOCP) | CVXPY with Clarabel or HiGHS | cuOpt QP (beta) | |
| Modelling layer | Pyomo or CVXPY; plain highspy for cut loops | | |

Size thresholds, from verified runs: an exact two-index CVRP MIP with greedy
capacity cuts **proves optimality up to ~30 customers in seconds** (A-n32-k5
in 13 s alone). At 100 customers it leaves a ~5-6% certified gap. Closing that
needs branch-cut-and-price, which no open-source package offers ready-made.

## Baseline first (mandatory)

Before any production or quantum claim, reproduce a known answer on the same
code path:

- CVRP: `E-n22-k4` (optimum 375) and `A-n32-k5` (optimum 784) must come out
  proven optimal; `X-n101-k25` must reach its best-known 27591.
- A heuristic result is reported with a **lower bound** from HiGHS, never as
  "optimal".

```bash
./run_all.sh     # anchors, Chicago race, QAOA readiness test (~10 min on 4 vCPU)
python race.py data/A-n32-k5.vrp --time 60 --bks 784 --out results/a-n32-k5.json
```

| Instance | Known | Best found | Lower bound | Status |
|---|---|---|---|---|
| E-n22-k4 | 375 | 375 | 375 | proven optimal |
| A-n32-k5 | 784 | 784 (PyVRP, <1 s) | 784 | proven optimal |
| X-n101-k25 | 27591 | 27591 (PyVRP, 30 s) | 26034 | certified gap 5.6% |
| Chicago, 80 stops, OSM roads | none | 145.7 km (PyVRP) | 138.2 km (directed bound, `bound.py`, 900 s) | certified gap 5.15% |

## Benchmark: how good is it? (CVRPLIB X, verified 2026-10-01)

The field's bar is the published mean gap to best-known solutions at
Tmax = 2.4·n s: HGS-CVRP 0.11%, PyVRP 0.22%. Measured on 30 stratified
instances (n 100-1000, `data/X/xsub.txt`), one seed, at the **full published
budget** (one pinned core per instance, 54 min wall for all 30 on a 32-vCPU
instance):

| Size | PyVRP | Race (PyVRP + cuOpt) |
|---|---|---|
| n 100-299 (13) | 0.15% | **0.14%** (beats the bar) |
| n 300-599 (10) | 0.30% | 0.27% |
| n 600-1000 (7) | 0.74% | 0.69% |
| all 30 | 0.34% | **0.31%** |

Verdict: **close, not reached** (bar 0.22%). The gap is on the largest instances;
one seed and an uncalibrated CPU speed add noise. At 10% of the budget the race
gives 0.49% (PyVRP 0.60%, cuOpt on an L4 0.99%, OR-Tools 7.96%, with no plan on 4 tight instances). All 100
instances at 3 seeds costs about $5 on `cpu-32v-128g`; `fullbench.py --cores <range>`
runs one instance per pinned core, longest first. See `fleet-routing/README.md`
for every number.

## Traps (all hit for real)

- **cuOpt minimises fleet size first, then distance.** On distance-only
  benchmarks its default lost 13% on X-n101. Setting `min_vehicles = k_min + 1`
  reached the best-known solution; an explicit zero fleet cost gave +0.2%. Choose
  the setting on one instance, then freeze it. For real fleet-cost problems the
  default is the right one.
- **Asymmetric (road) distances: bound the directed model.** A symmetric
  `min(d_ij, d_ji)` relaxation is valid but loose: 8.4% against 5.15% on Chicago.
- **`highspy` 1.15 and `ortools` 9.15 cannot be imported into the same Python
  process.** Each one's bundled HiGHS breaks the other's shared library
  (undefined symbol at import). Run them in separate processes (the racing
  harness does this).
- **`overpass-api.de` returns HTTP 406 to cloud IPs.** Set `OVERPASS_URL` to
  another public Overpass mirror (the example used
  `maps.mail.ru/osm/tools/overpass/api/interpreter`), a self-hosted Overpass,
  or use a city open-data portal for buildings. Check the user's data-sourcing
  policy before picking a third-party mirror.

## Compute

- Up to a few hundred customers and the known-answer checks: the subscription pod
  or a `cpu-8v-32g` instance.
- Benchmarks and long races: `cpu-32v-128g`, one instance per pinned core.
- cuOpt: `gpu-l4` ($0.49/h) or `gpu-h100-sxm` for very large instances.
- Launch through **qbraid-cloud-orchestration** with `--auto-stop`, copy results
  back, and terminate when done. State the estimate and get the user's OK before launching.

## Quantum: the short answer

**Runnable test today.** QUBO/QAOA fits sub-problems of about 16-20 binary
variables, as a *readiness experiment* scored against the brute-force or MIP
optimum, never as an advantage claim. Measured here in noiseless simulation
(`qaoa_tsp.py`, one 4-stop route, 16 qubits, p = 1-3, 1000 shots): P(optimal) 0.11-0.14%, which is 70-90x
better than a random bitstring but 30-40x *below* picking a random valid tour,
because 97% of shots break the one-hot constraints. The next step is a
constraint-preserving (XY) mixer or tuned penalties, in simulation, before buying
QPU time; hardware will do worse than this noiseless figure. Price every route
with `qbraid devices get <qrn>` and confirm with the user (credits; 1 credit =
$0.01). Up to 20 binaries, IQM Garnet (`aws:iqm:qpu:garnet`): 30 per task + 0.145
per shot, so p = 1-3 at 1000 shots each is 525 credits ($5.25). Up to about 50,
IQM Emerald at 0.16 per shot. Dense QUBOs without SWAPs, IonQ Forte
(`aws:ionq:qpu:forte-1`, 36 qubits, all-to-all) costs 8 per shot: the same three
runs are 24,090 credits ($240.90), so price it before suggesting it. Larger
problems: IBM Heron (156 qubits, billed to the user's IBM account). No quantum
annealer is online on qBraid today.

For QAOA on IBM hardware and QOBLIB benchmarking, start from `quantum-optimization/`
in the same repository (environment `qiskit_zngl3z`). For the full answer and the experiment protocol, use the **quantum-readiness**
skill (section "Optimization").

## Verification stamp

- 2026-10-01: PyVRP 0.14.0, OR-Tools 9.15.6755, highspy 1.15.1, Qiskit 2.5.2, Python 3.12 (`requirements.txt`); cuopt-cu12 26.8.0 on a `gpu-l4` instance.
- Full-budget CVRPLIB X run: 30 instances, 10 pinned cores of a `cpu-32v-128g` instance, 54 min wall, 8.3 CPU-hours.
- Anchors, Chicago race and QAOA: the subscription pod, 3 processes x 1 thread, 60 s per race, 0 credits.
- Chicago directed bound: 900 s on 2 threads.
- Re-verify on any PyVRP, OR-Tools, HiGHS or cuOpt version bump.
