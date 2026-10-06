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
- `skills/` holds draft agent skills distilled from the verified runs. They are being published through the qBraid skills registry (`qbraid skills search <topic>`); until then, read them here.
- `docs/pathway.html` is the sector survey these examples came from: open-source alternatives by sector, their licences, and where quantum fits.

## Third-party data and licences

The code here is Apache-2.0. Some examples bundle or derive from third-party data, which keeps its own terms:

- **Fleet routing:** CVRP instances and best-known solutions from [CVRPLIB](http://vrp.galgos.inf.puc-rio.br/) (Uchoa et al. 2017 X set; Christofides/Augerat E and A sets), redistributed for benchmarking with citation. Chicago street network and buildings © OpenStreetMap contributors, [ODbL](https://www.openstreetmap.org/copyright).
- **Energy grids:** test cases are fetched at run time from [PGLib-OPF](https://github.com/power-grid-lib/pglib-opf) (CC-BY-4.0); the SciGrid-DE network ships with PyPSA.
- **Weather:** contains modified Copernicus Climate Change Service information (ERA5), 2026; neither the European Commission nor ECMWF is responsible for any use of it. IFS HRES and ensemble reference scores come from [WeatherBench 2](https://sites.research.google/weatherbench/). Coastlines from [Natural Earth](https://www.naturalearthdata.com/) (public domain).
- **Drug discovery:** protein structures from the [RCSB PDB](https://www.rcsb.org/) (CC0). Benchmark complexes, and per-complex Vina and GOLD results, from the PoseBusters benchmark (Buttenschoen et al. 2024). FEP+ reference predictions from Schrödinger's public FEP+ benchmark repository (MIT).
- **Materials:** structures from the [Materials Project](https://next-gen.materialsproject.org/) (CC-BY-4.0) and the WBM set via [Matbench Discovery](https://matbench-discovery.materialsproject.org/) (CC-BY-4.0 data, MIT code). Experimental lattice constants and bulk moduli from Csonka et al. (2009).
- **Finance:** Ken French Data Library and CBOE index history are downloaded at run time by `finance-risk/fetch_data.sh` and are not redistributed here.
- **Robotics:** Unitree Go1 model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (BSD-3-Clause).
- **Topology optimization:** the reference codes `top88.m` (Andreassen et al. 2011) and `top3d.m` (Liu and Tovar 2014) are fetched at run time and not redistributed; only a headless driver is included.
- **Qubit design:** the transmon example and regression reference come from [AWS Palace](https://github.com/awslabs/palace) (Apache-2.0).

## License

Apache-2.0. The solvers keep their own licences (OpenFOAM GPL-3.0, Palace
Apache-2.0, HiGHS MIT, cuOpt Apache-2.0, PyVRP MIT).
