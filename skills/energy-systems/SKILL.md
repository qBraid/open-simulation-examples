---
name: energy-systems
description: Run power-system studies on qBraid with open-source tools: AC optimal power flow (Egret/Pyomo + Ipopt, PGLib-OPF benchmarks), power flow and grid studies (pandapower), and capacity-expansion and dispatch planning on real networks (PyPSA + HiGHS). Use when a user brings an OPF, unit-commitment, grid-expansion, renewable-integration or energy-planning problem, or asks for a PSS/E, PowerWorld or PLEXOS alternative. Gives the tool decision rules, the PGLib known-answer check, the certification step (SOC lower bound), verified costs and the gotchas.
metadata:
  version: "0.3.0"
  status: "provisional"
  verified: "2026-10-01"
---

# Energy systems on qBraid

Open-source power-system tools, picked by question, checked against PGLib or a
published reference before any claim. Pair with **solver-racing** for the LP/MIP
layer and with **qbraid-cloud-orchestration** for instances.

Start from the user's goal: if it is unclear whether they want the best solution
or the best quantum solution, ask (see solution-router).

## Get the scripts

The worked example lives in `energy-grid/` of
https://github.com/qBraid/open-simulation-examples:

```bash
git clone --depth 1 https://github.com/qBraid/open-simulation-examples
cd open-simulation-examples/energy-grid
```

All script references below are relative to that directory. `run_all.sh` chains
the steps.

## Environment

Install the qBraid environment (kernel included): `qbraid envs install energy_5zvlz6`
(**energy-systems**: Pyomo 6.10.1, Egret 0.6.2, PyPSA 1.2.4, HiGHS 1.15.1, pandapower 3.5.5,
and a bundled Ipopt 3.14.20 that Pyomo finds with no PATH setup). Verified from a fresh
install on 2026-10-06: case118_ieee and case300_ieee match PGLib to 4e-6. Without qBraid,
build it with conda-forge (Ipopt needs it):

```bash
# micromamba is not in the Lab image: fetch the static binary first
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /tmp bin/micromamba
export PATH=/tmp/bin:$PATH
export MAMBA_ROOT_PREFIX=/tmp/mamba ENV=/tmp/envs/energy   # local disk; /tmp resets on restart
micromamba create -y -p $ENV -c conda-forge python=3.12 ipopt=3.14 pyomo pypsa highspy networkx scipy pandas numpy netcdf4
$ENV/bin/pip install gridx-egret "pandapower>=3"
export PATH=$ENV/bin:$PATH OMP_NUM_THREADS=1   # Pyomo finds the ipopt binary on PATH only
git clone --depth 1 https://github.com/power-grid-lib/pglib-opf.git
```

To reuse it across instances, package it as a qBraid environment with the
**manage-environments** skill.

## Pick the tool (decision rules)

| Question | First choice | Notes |
|---|---|---|
| AC optimal power flow, transmission | Egret (Pyomo) + Ipopt | Validated against PGLib-OPF v23.07 (below). Race polar, rectangular and current-voltage formulations: they agree to 8 digits when all converge, and the race protects against a formulation that stalls on a given case. |
| Certify an AC-OPF answer | SOC relaxation (Egret `create_soc_relaxation(..., use_linear_relaxation=False)`) + Ipopt | Convex, so Ipopt's optimum is a valid lower bound. Gap = (AC - SOC)/AC. Reproduces PGLib's published SOC gaps. A local solver alone never proves optimality; this does (up to the gap). |
| Power flow, distribution, contingency screening | pandapower >= 3 | Needs pandapower 3.x with numpy 2 (2.x imports `numpy.Inf` and fails). |
| Dispatch and capacity expansion (lines, storage, generation) | PyPSA + HiGHS | Linear (DC) network. Follow with `n.pf()` for AC voltages. |
| Very large LPs (multi-year, hourly, continental) | PyPSA + HiGHS IPM, raced against simplex; cuOpt PDLP on GPU above ~10^6 nonzeros | See solver-racing. |
| Unit commitment | PyPSA (committable) or Egret UC + HiGHS MIP | MIP: HiGHS is behind commercial solvers (see optimization-stack). |

PSS/E, PowerWorld and PowerFactory still lead on dynamic and transient
stability, protection, and vendor-certified device models. Don't claim
replacement for those. PLEXOS's edge is integrated market data and
stochastic unit commitment at scale. PyPSA matches it on the modelling, but the
data pipeline is the user's to build.

## Baseline first (mandatory)

Run before any claim, on the same code path:

- **AC-OPF:** `pglib_opf_case118_ieee` must give 97,214 $/h (PGLib reference 9.7214e+04) with
  a certified gap of about 0.90% (PGLib's SOC gap 0.91%). `case300_ieee` must give 565,220 (reference 5.6522e+05, SOC gap 2.62% vs 2.63%).
- **Expansion:** the PyPSA SciGrid-DE dispatch (`pypsa.examples.scigrid_de()`, with the example's preparation)
  solves in under 20 s and its AC power flow converges in 24/24 hours (with the PV-control recipe below).

```bash
python opf/acopf_pglib.py pglib-opf out case118_ieee case300_ieee case1354_pegase case2869_pegase case118_ieee__api
python -c "import pypsa; pypsa.examples.scigrid_de().export_to_netcdf('scigrid_de.nc')"
python expansion/expansion.py scigrid_de.nc exp w1.0_s1.0_c0 --pf
```

| Case | Buses | Gap to PGLib ref | Certified gap (ours / PGLib SOC) | Fastest formulation |
|---|---|---|---|---|
| case118_ieee | 118 | -4e-6 | 0.90% / 0.91% | 0.3 s |
| case1354_pegase | 1,354 | +3.5e-5 | 1.57% / 1.57% | 4.1 s |
| case2869_pegase | 2,869 | -3.9e-6 | 1.01% / 1.01% | 17.5 s |
| case3012wp_k | 3,012 | +1.6e-5 | 1.02% / 1.03% | 8.9 s |
| case118_ieee__api (stress) | 118 | +1.8e-5 | 26.16% / 26.17% | 0.3 s |

All 16 cases attempted (10 TYP, 3 API, 3 SAD; 14 to 3,012 buses) are within 5e-5
of the reference. The full table is in `energy-grid/README.md`.

## Gotchas (all hit for real)

- **PGLib's `BASELINE.md` writes `inf.` in the DC column for the SAD cases.** Parse each field independently or those rows are silently dropped.
- **Pyomo can't find Ipopt** unless the env's `bin` is on `PATH` (`No executable found for solver 'ipopt'`).
- **Out-of-service branches** come back from Egret with `pf = None`. Skip them when exporting flows (case300).
- **PyPSA AC power flow after an LOPF diverges** on SciGrid-DE (NaN voltages in most hours) unless all
  generators are set to `control = "PV"`, with the units at one bus set to `PQ`. This is PyPSA's own
  lopf-then-pf recipe. Keep the LOPF dispatch for the generation mix: the PF slack absorbs the losses.
- **Race simplex against IPM for expansion.** SciGrid-DE dispatch solves in 14-17 s. With extendable lines and a
  candidate battery at every bus, dual simplex alone took 766 s, while HiGHS IPM won the race in 58-124 s on all 4 scenarios (about 10x faster).
- **One representative day undervalues storage.** With annualised costs scaled to 24 h, batteries at
  EUR 120k/MW/yr are not built in the base case. Use multi-day or clustered periods before drawing storage conclusions.
- **MUMPS vs HSL:** PGLib's timings use HSL MA27, which is 2-6x faster than MUMPS. HSL has a separate licence, so compare our times with that in mind.

## Compute

- PGLib up to ~3,000 buses and a one-day SciGrid-DE expansion run single-threaded
  in seconds to minutes: the subscription pod or a `cpu-8v-32g` instance is enough.
- Scenario ensembles and multi-day expansion: one scenario per core on
  `cpu-32v-128g`, launched through **qbraid-cloud-orchestration** with `--auto-stop`,
  terminated when done. State the estimate and get the user's OK before launching.

## Quantum: the short answer

**Research-stage test only.** Unit commitment and network expansion fit QUBO/QAOA
only at toy size; the LP/NLP stack above wins by orders of magnitude, and the SOC
relaxation gives a certificate no sampler does. The honest experiment is a small
unit-commitment or line-selection sub-problem (under ~20 binaries) run through the
optimization readiness test (QAOA in simulation, then a priced QPU run after the
user confirms), scored against the exact MIP optimum, with the SOC bound as the
classical certificate. Power-system dynamics is not a near-term quantum target.

For the full answer and the experiment protocol, use the **quantum-readiness**
skill (section "Optimization").

## Verification stamp

- 2026-10-01: conda-forge Python 3.12, Ipopt 3.14.20 (MUMPS), Pyomo 6.10.1, gridx-egret, PyPSA 1.2.4, highspy, pandapower 3.5.5, numpy 2.4.6.
- Machine: CPU only, 1 thread per solve, on a `gpu-l4` instance (about 5 CPUs by cgroup).
- AC-OPF: PGLib-OPF v23.07 (commit dc6be4b), 16 cases, 3 formulations raced plus an SOC relaxation each: about 25 min wall in total.
- Expansion: PyPSA SciGrid-DE (2011-01-01, 24 h), 4 scenarios, 58-124 s per scenario (IPM won all 4).
- Cost: about 0.8 CPU-slot-hours, roughly $0.20.
