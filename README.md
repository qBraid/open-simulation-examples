# Open Simulation Examples

Engineering-grade simulation with open-source solvers, set up and run by agents
on qBraid compute, and shown in the browser with three.js.

Each example is a complete workflow. It starts from a problem statement, runs a
solver on qBraid CPU or GPU instances, checks the result against a known answer,
and renders an interactive viewer in the qBraid Agent Canvas. It also shows where
quantum computing fits today and where it could fit in the future.

| Example | Open-source stack | Stands in for | Quantum angle |
|---|---|---|---|
| [`wind-tunnel/`](wind-tunnel/) | OpenFOAM, gmsh, XLB | Ansys Fluent, STAR-CCM+ | none practical today; stated plainly |
| [`fleet-routing/`](fleet-routing/) | NVIDIA cuOpt, PyVRP, HiGHS, OR-Tools | Gurobi, CPLEX routing | QAOA on a sub-problem, scored against the exact optimum |
| [`qubit-design/`](qubit-design/) | AWS Palace, gmsh, Qiskit Metal, CUDA-Q | Ansys HFSS | design a transmon, then simulate and compare with real devices |

Every result in this repo carries a verification stamp: the date, environment,
instance profile, wall time and cost.

## Layout

- `<example>/README.md` gives the problem, how to run it, the verified results and the known limits.
- `<example>/results/` holds small JSON results that the viewers read.
- `<example>/viewer.html` is a self-contained three.js viewer.
- `skills/` holds draft agent skills distilled from the verified runs.
- `docs/pathway.html` is the sector survey these examples came from.

## License

Apache-2.0. The solvers keep their own licences (OpenFOAM GPL-3.0, Palace
Apache-2.0, HiGHS MIT, cuOpt Apache-2.0, PyVRP MIT).
