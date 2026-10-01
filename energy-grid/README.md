# Power grids: AC optimal power flow and capacity expansion

Open-source power-system optimisation, run by agents on qBraid compute, checked
against the community benchmark, and shown in a three.js "Power Grid Studio".

| | Open-source stack | Stands in for |
|---|---|---|
| AC optimal power flow | Egret (Pyomo) + Ipopt 3.14, pandapower | PSS/E OPF, PowerWorld OPF |
| Dispatch and capacity expansion | PyPSA + HiGHS | PLEXOS (LP planning core) |

![Germany, dark theme](results/viewer_grid_dark.png)

## What top-10% means here, and the verdict

**AC-OPF.** The community benchmark is [PGLib-OPF](https://github.com/power-grid-lib/pglib-opf) (IEEE PES
task force). Its `BASELINE.md` publishes the best-known AC objective for every case, from
PowerModels.jl + Ipopt, the de-facto research reference, together with the SOC and QC
relaxation gaps. Most published new methods (including ML-based OPF) report optimality gaps
in the 0.1–1% range. Our bar:

1. Match the published AC objective **within 0.1%** on every case attempted, including PGLib's
   stress variants (API: active-power increase; SAD: small angle difference).
2. **Certify** each answer ourselves. A local NLP solver only finds a local optimum, so
   we also solve the convex SOC relaxation, which gives a lower bound, and report the certified gap.
   It should reproduce PGLib's published SOC gap.

**Verdict: __OPF_VERDICT__**

__OPF_TABLE__

**Capacity expansion.** No public leaderboard exists for planning models. The bar we
can defend is method parity:
- Reproduce PyPSA's documented SciGrid-DE study (the real German grid) end to end.
- Add co-optimised line and storage expansion.
- Race the LP solvers.
- Validate the dispatch with a full AC power flow every hour.

__EXP_VERDICT__

__EXP_TABLE__

## The viewer

`viewer.html` is self-contained: the data is inlined, and three.js 0.170 loads from jsDelivr.

- **Germany · 24 h.** The 585-bus SciGrid-DE network on a map:
  - **Flow particles** on every line; density and speed follow the MW, direction follows the flow.
  - **Line colour:** loading against the N-1 limit (viridis). Congested lines (≥ 98%) pulse.
  - **Bus voltages** from the AC power flow, as pillars and halos.
  - **New line capacity and batteries** chosen by the optimiser.
  - A **24-hour scrubber** with an animated stacked generation mix, a **scenario switcher** with deltas against the base, and a data-driven **guided tour**.
- **AC-OPF benchmark.** The IEEE 118, PEGASE 1354 and PEGASE 2869 networks in 3D:
  - **Bus height** is the solved voltage, between glass planes at the legal limits.
  - **Buses pinned at a limit** glow red: these are the binding constraints that set prices.
  - **Panels:** the gap to PGLib, the certified gap against PGLib's SOC gap, and the formulation race.

URL hash options: `#dark`, `#light`, `#opf`, `#tour`, combinable (`viewer.html#opf-dark`).

| | |
|---|---|
| ![grid light](results/viewer_grid_light.png) | ![tour](results/viewer_tour_dark.png) |
| ![opf 118](results/viewer_opf118_light.png) | ![opf 2869](results/viewer_opf2869_dark.png) |

## Reproduce

```bash
# on a qBraid instance (here: the shared L4 pool box, ~5 cgroup CPUs)
export MAMBA_ROOT_PREFIX=/tmp/ose/mamba ENV=/tmp/ose/energy-grid/env
micromamba create -y -p $ENV -c conda-forge python=3.12 ipopt=3.14 pyomo pypsa highspy networkx scipy pandas numpy netcdf4
$ENV/bin/pip install gridx-egret "pandapower>=3"
export PATH=$ENV/bin:$PATH OMP_NUM_THREADS=1
git clone --depth 1 https://github.com/power-grid-lib/pglib-opf.git

# AC-OPF: race 3 formulations, compare with BASELINE.md, certify with SOC
python opf/acopf_pglib.py pglib-opf out case118_ieee case1354_pegase case2869_pegase case118_ieee__api ...

# Expansion: SciGrid-DE (python -c "import pypsa; pypsa.examples.scigrid_de().export_to_netcdf('scigrid_de.nc')")
python expansion/expansion.py scigrid_de.nc exp w1.0_s1.0_c0 --pf      # wind x1, solar x1, CO2 0 EUR/t

# Viewer (on the pod; networkx for layouts)
python build_viewer.py
```

`run_all.sh` chains these steps.

## Honest limits

- **Not a PSS/E replacement for dynamics.** Transient and small-signal stability, protection
  coordination and vendor-certified device models are out of scope for these tools.
- **The PLEXOS comparison is about the optimisation core, not the product.** PLEXOS's integrated market data,
  stochastic unit commitment and support are not reproduced here.
- **Expansion uses a linear (DC) network with fixed impedances,** like every LP planning model. We check the
  dispatch with a full AC power flow afterwards. Line upgrades are treated as capacity only.
- **One representative day (2011-01-01) and annualised costs scaled to 24 h.** This is a method demonstration,
  not an investment recommendation. Storage in particular needs multi-day periods to be valued fairly.
- **The voltage set points in SciGrid-DE are unknown.** Generators are voltage-controlled at 1.0 p.u. (PyPSA's recipe),
  so the bus voltages show the network's response, not measured values.
- **Timings are on a shared box with about 5 CPUs and Ipopt with MUMPS.** PGLib's reference timings use HSL MA27, which is 2–6× faster.

## Quantum, honestly

- **Today:** unit commitment and expansion fit QUBO/QAOA only at toy size. The LP and NLP stack above wins by
  orders of magnitude and gives certificates (SOC gaps) that no sampler does.
- **Later:** this is optimisation, so the outlook is as in `skills/optimization-stack`: Grover-type speedups need fault
  tolerance and are largely eaten by overheads at practical sizes.

## Verification stamp

__STAMP__
