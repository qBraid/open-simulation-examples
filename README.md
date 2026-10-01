# Open Simulation Examples

Engineering-grade simulation with open-source solvers, set up and run by agents
on qBraid compute, and shown in the browser with three.js.

Each example is a complete workflow. It starts from a problem statement, runs a
solver on qBraid CPU or GPU instances, checks the result against a known answer,
and renders an interactive viewer in the qBraid Agent Canvas. It also shows where
quantum computing fits today and where it could fit in the future.

| Example | Open-source stack | Stands in for | Benchmark bar (top-10% for the field) | Verdict |
|---|---|---|---|---|
| [`wind-tunnel/`](wind-tunnel/) | OpenFOAM v2412, gmsh | Ansys Fluent, STAR-CCM+ | Cylinder Re=100 St/Cd bands; Ahmed body drag vs experiment | Cylinder **reached**; Ahmed 35° **close** (0.4% off, mesh not settled); Ahmed 25° **not reached** (steady-RANS limit) |
| [`fleet-routing/`](fleet-routing/) | PyVRP, NVIDIA cuOpt, HiGHS, OR-Tools | Gurobi/CPLEX routing, Hexaly | CVRPLIB X: PyVRP's published 0.22% mean gap to best-known | **Close**: 0.31% at the full budget (30-instance subset); beats the bar below 300 customers |
| [`qubit-design/`](qubit-design/) | AWS Palace, gmsh | Ansys HFSS | Modern transmon Hamiltonian; predicted T1 inside IBM medians | **Reached** (3D-verified: 4.81 GHz, -270 MHz, T1 142–250 µs predicted); Ta-record coherence not reached |
| [`drug-discovery/`](drug-discovery/) | Boltz-2, RDKit, PoseBusters | Schrödinger Glide, FEP+ | Co-folding pose success vs AlphaFold3/Chai; affinity ranking vs FEP+ | Poses **reached** (84% on a filtered subset); affinity **close** to FEP+ (Spearman 0.77 / 0.87) |
| [`topology-optimization/`](topology-optimization/) | Python ports of top88/top3d, scikit-sparse | Altair OptiStruct, Abaqus Tosca | Match the published reference codes within 1% | **Reached** (to ~1e-5%); printable titanium bracket, 88% lighter |
| [`robotics/`](robotics/) | MuJoCo Playground, Brax, MuJoCo WASM | MATLAB Simulink, proprietary sims | Published Playground Go1 joystick reward and tracking | **Reached** (reward 28.1 vs ~27; 0 falls); live in-browser policy |
| [`energy-grid/`](energy-grid/) | Egret/Pyomo + Ipopt, PyPSA, HiGHS | Siemens PSS/E, PLEXOS | PGLib-OPF best-known objectives within 0.1% | **Reached** (16 cases to 3,012 buses within 5e-5) |
| [`weather/`](weather/) | NVIDIA Earth2Studio, FourCastNet 3 | Operational NWP (ECMWF IFS) | WeatherBench2 RMSE vs IFS HRES, days 1–5 | **Close**: FCN3 4-member mean beats HRES on 7 of 9 pairs; day-3 z500 5% worse |
| [`finance-risk/`](finance-risk/) | QuantLib, CuPy, Riskfolio-style backtest | Bloomberg PORT, Numerix, MSCI | Basis-point pricing; MC error bars match theory; DeMiguel-protocol backtest | **Reached** on all three (backtest reproduces "1/N is hard to beat") |
| [`materials/`](materials/) | MACE, ORB v3, EquiformerV3, PySCF | VASP, Materials Studio | Matbench Discovery top-10% (F1 ≥ 0.925) | **Reached** via the #1 MIT-licensed model (F1 0.909 on 250, matches its 0.931 entry); phonons not reached |

All verdicts are relative to published references, stated in each README with the date, environment, machine and cost. "Close" means a specific, costed next step would decide it.

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
