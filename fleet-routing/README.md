# Fleet routing on a live map

Deliver to 80 stops on the real Chicago road network with a capacity-limited
fleet, using open-source solvers **raced** against each other. The race's best
answer is certified with a lower bound, so "good" becomes "at most X% from
optimal". The result renders as a three.js city with vehicles driving their
routes.

It stands in for commercial routing and MIP stacks (Gurobi, CPLEX and
routing products built on them). It also shows, with measurements, where
quantum optimisation stands today.


## What runs

| Piece | Tool | Role |
|---|---|---|
| Hybrid genetic search | [PyVRP](https://github.com/PyVRP/PyVRP) 0.14 (MIT) | state-of-the-art CVRP heuristic |
| Guided local search | [OR-Tools](https://github.com/google/or-tools) 9.15 routing (Apache-2.0) | the most widely used open routing library |
| Exact MIP + cuts | [HiGHS](https://github.com/ERGO-Code/HiGHS) 1.15 via highspy (MIT) | two-index CVRP with rounded-capacity cuts; **lower bounds** and proofs |
| Road network | [OSMnx](https://github.com/gboeing/osmnx) 2.1 on OpenStreetMap | real driving distances, one-way aware |
| QAOA | NumPy statevector, cross-checked with Qiskit 2.5 | readiness experiment on one route |
| Viewer | three.js 0.170 | 3D city, animated vehicles, convergence and QAOA panels |

Every solver runs in its own process with the same time budget (`race.py`).
That is the local form of racing across qBraid instances, one solver per
machine. It is also required: `highspy` 1.15 and `ortools` 9.15 cannot be
imported into the same Python process (their bundled HiGHS symbols clash).

## Reproduce

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
./run_all.sh                 # ~10 min on 4 vCPU; writes results/*.json and viewer.html
qbraid-canvas viewer.html --title "Fleet routing"   # or open viewer.html in a browser
```

Single pieces:

```bash
python race.py data/A-n32-k5.vrp --time 60 --bks 784 --out results/a-n32-k5.json
python race.py data/X-n101-k25.vrp --time 60 --certify 240 --bks 27591 --out results/x-n101-k25.json
python city.py --stops 80 --time 60          # fetches OSM, races, stores street geometry
python qaoa_tsp.py                           # QAOA on one route of the city solution
python build_viewer.py
```

## Results

Verification stamp: **2026-09-30**, qBraid subscription pod (shared, 8 vCPU
tier, heavily loaded by other jobs), 3 processes × 1 thread, 60 s race budget,
**0 credits**, env `requirements.txt`.

### Known-answer anchors (CVRPLIB)

| Instance | Customers | Known | PyVRP | OR-Tools | HiGHS lower bound | Status |
|---|---|---|---|---|---|---|
| E-n22-k4 | 21 | 375 (optimal) | 375 at <1 s | 375 at <1 s | 375 | **proven optimal** |
| A-n32-k5 | 31 | 784 (optimal) | 784 at <1 s | 796 at 60 s (784 at 120 s in a longer run) | 784 | **proven optimal** |
| X-n101-k25 | 100 | 27591 (BKS) | 27591 at 30 s | 29159 (+5.7%) at 60 s | 26034 after a 240 s certify step | certified gap 5.6% |

Gap to the best-known answer at fixed time limits (X-n101-k25):

| Solver | 1 s | 5 s | 10 s | 30 s | 60 s |
|---|---|---|---|---|---|
| PyVRP | +1.34% | +0.25% | +0.14% | **0.00%** | 0.00% |
| OR-Tools | +9.31% | +5.68% | +5.68% | +5.68% | +5.68% |

HiGHS proves A-n32-k5 on its own in 13 s when it has the machine to itself, and in
about 30 s alongside the other two racers. At 100 customers the simple cut loop
leaves a 5.6% certified gap. Published proofs at that size use
branch-cut-and-price, which the open stack doesn't have in a ready-made form.

### Chicago, 80 stops, real roads

| Solver | 1 s | 10 s | 30 s | 60 s | Vehicles |
|---|---|---|---|---|---|
| PyVRP | 146.7 km | **145.7 km** | 145.7 km | 145.7 km | 13 |
| OR-Tools | 155.1 km | 153.1 km | 151.4 km | 151.2 km (+3.8%) | 13 |
| HiGHS | none | none | none | bound 133.4 km | none |

Winner: PyVRP, 145.7 km. **Certified gap 8.4%**: no plan can be shorter than
133.4 km. The bound comes from a symmetric relaxation of one-way distances, so
it is looser than on the symmetric benchmarks. 1,801 intersections, 4,341
street segments, capacity 100, demands 5–25, seed 7.

### Quantum readiness: QAOA on one route

One vehicle's route from the city solution (4 stops; no route had exactly 4,
so this is the first 4 stops of the shortest route), with the depot fixed as the
start and end. It is written as a stop × position QUBO on **16 qubits** with
row and column penalties (A = 8 on distances normalised to 1). QAOA runs on an
exact statevector at depth p = 1, 2, 3 (COBYLA; INTERP initialisation from
p to p+1). It is scored by **1000 drawn shots**, as a QPU run would be. The
same circuit built in Qiskit (angles bound by name) reproduces the state with
overlap 1.000000 at every depth.

| p | P(optimal tour) | P(valid tour) | optimal in 1000 shots | best valid tour in shots |
|---|---|---|---|---|
| 1 | 0.110% | 2.6% | 3 | 8.30 km (optimal) |
| 2 | 0.126% | 3.0% | 0 | 8.31 km (+0.06%) |
| 3 | 0.140% | 3.3% | 2 | 8.30 km (optimal) |

The classical anchor: brute force over all 24 tours takes 0.1 ms and gives
8.30 km (the worst tour is 11.89 km). PyVRP's order for this route was already
optimal.

**How to read it:** QAOA puts 70–90x more probability on the optimum than a
uniformly random bitstring (0.0015%). But it puts 30–40x *less* than simply
picking a random valid tour (4.2%), because 97% of shots violate the one-hot
constraints. Deeper circuits help only slowly. That is the known weakness of
penalty-encoded QUBOs, and it is the thing to beat before hardware time is
worth buying: constraint-preserving mixers (XY / permutation), better penalty
tuning, or warm starts. Nothing here suggests an advantage. It shows that the
pipeline (encoding, angle transfer and shot-based scoring) is correct and ready
to point at a QPU.

## Honest positioning

- **Routing:** at parity with commercial stacks for practical use. PyVRP reached
  every known answer tested here.
- **General MIP:** not Gurobi-class. On the Mittelmann MIPLIB2017 benchmark
  (8 threads, 27 Apr 2026) HiGHS is 7.4x slower than the fastest commercial
  solver and solves 68% of instances against 91%. Racing, warm starts and
  decomposition narrow the gap; they do not close it.
- **LP:** the GPU solver cuOpt ranked first on Mittelmann LPfeas (12 Jun 2026).
- **Quantum:** a readiness experiment on a 16-qubit sub-problem, not an advantage.

## Next: needs an instance

The account was at its concurrent-instance limit during this run, so these
are not yet verified.

| Step | Profile | Estimate |
|---|---|---|
| cuOpt as a fourth racer (VRP + LP), `requirements-gpu.txt` | `gpu-l4` ($0.49/h) | ~1 h, **~$0.50** |
| cuOpt at scale (1,000+ stops, LP relaxations) | `gpu-h100-sxm` ($5.37/h) | ~1 h, **~$5.40** |
| One racer per instance (PyVRP / OR-Tools / HiGHS on separate `cpu-8v-32g`) | 3 × $0.48/h | ~30 min, **~$0.75** |
| QAOA sampled on a QPU, p = 1–3, 1000 shots each | IQM Garnet: 30 credits/task + 0.145/shot | **525 credits ($5.25)**. Expect mostly noise: 16 qubits with dense penalty couplings need many SWAPs. Ask before spending. |

## Files

- `vrp/instance.py`: loading, TSPLIB distances, solution checking
- `vrp/solvers.py`: the three solvers with a common interface
- `race.py`: racing harness, gap tables, certify step
- `city.py`: OSM scenario and street geometry
- `qaoa_tsp.py`: TSP QUBO, QAOA simulator, Qiskit cross-check
- `build_viewer.py` + `viewer_template.html` produce `viewer.html` (self-contained, data inlined)
- `data/`: CVRPLIB instances (E-n22-k4 from the PyVRP test data, A-n32-k5 from
  the CVRPLIB set A, X-n101-k25 from `PyVRP/Instances`)
