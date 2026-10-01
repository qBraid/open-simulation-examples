---
name: energy-systems
description: Run power-system studies on qBraid with open-source tools: AC optimal power flow (Egret/Pyomo + Ipopt, PGLib-OPF benchmarks), power flow and grid studies (pandapower), and capacity-expansion and dispatch planning on real networks (PyPSA + HiGHS). Use when a user brings an OPF, unit-commitment, grid-expansion, renewable-integration or energy-planning problem, or asks for a PSS/E, PowerWorld or PLEXOS alternative. Gives the tool decision rules, the PGLib known-answer check, the certification step (SOC lower bound), verified costs and the gotchas.
metadata:
  version: "0.1.0"
  layer: "1"
  status: "draft"
  verified: "2026-10-01"
---

# Energy systems on qBraid

Open-source power-system tools, picked by question, checked against PGLib or a
published reference before any claim. Pair with **solver-racing** for the LP/MIP
layer and with **qbraid-cloud-orchestration** for instances. The worked example
is `energy-grid/` in qBraid/open-simulation-examples.

## Pick the tool (decision rules)

| Question | First choice | Notes |
|---|---|---|
| AC optimal power flow, transmission | Egret (Pyomo) + Ipopt | Validated here against PGLib-OPF v23.07 (table below). Race polar, rectangular and current-voltage formulations: they agree to 8 digits when all converge, and the race protects against a formulation that stalls on a given case. |
| Certify an AC-OPF answer | SOC relaxation (Egret `create_soc_relaxation(..., use_linear_relaxation=False)`) + Ipopt | Convex, so Ipopt's optimum is a valid lower bound. Gap = (AC − SOC)/AC. Reproduces PGLib's published SOC gaps. A local solver alone never proves optimality; this does (up to the gap). |
| Power flow, distribution, contingency screening | pandapower ≥ 3 | Needs pandapower 3.x with numpy 2 (2.x imports `numpy.Inf` and fails). |
| Dispatch and capacity expansion (lines, storage, generation) | PyPSA + HiGHS | Linear (DC) network. Follow with `n.pf()` for AC voltages. |
| Very large LPs (multi-year, hourly, continental) | PyPSA + HiGHS IPM, race against simplex; cuOpt PDLP on GPU above ~10⁶ nonzeros | See solver-racing. |
| Unit commitment | PyPSA (committable) or Egret UC + HiGHS MIP | MIP: HiGHS is behind commercial solvers (see optimization-stack). |

PSS/E, PowerWorld and PowerFactory still lead on dynamic and transient
stability, protection, and vendor-certified device models. Don't claim
replacement for those. PLEXOS's edge is integrated market data and
stochastic unit commitment at scale. PyPSA matches it on the modelling, but the
data pipeline is yours to build.

## Baseline first (mandatory)

Run before any claim, on the same code path:

- **AC-OPF:** `pglib_opf_case118_ieee` must give 97,214 $/h (PGLib reference 9.7214e+04) with
  a certified gap of about 0.90% (PGLib's SOC gap 0.91%). `case300_ieee` must give 565,220 (reference 5.6522e+05, SOC gap 2.62% vs 2.63%).
- **Expansion:** the PyPSA SciGrid-DE dispatch (`pypsa.examples.scigrid_de()`, with the example's preparation)
  solves in under 20 s and its AC power flow converges in 24/24 hours (with the PV-control recipe below).

## Verified recipe (stamp: 2026-10-01, qBraid L4 pool box, Ipopt 3.14.20 with MUMPS, 1 thread)

```bash
export MAMBA_ROOT_PREFIX=/tmp/ose/mamba
micromamba create -y -p $ENV -c conda-forge python=3.12 ipopt=3.14 pyomo pypsa highspy networkx scipy pandas numpy netcdf4
$ENV/bin/pip install gridx-egret "pandapower>=3"
export PATH=$ENV/bin:$PATH            # Pyomo finds the ipopt binary on PATH only
python acopf_pglib.py pglib-opf out case118_ieee case1354_pegase ...
```

| Case | Ours ($/h) | PGLib ref | Gap to ref | Certified gap (ours / PGLib SOC) | Wall time |
|---|---|---|---|---|---|
| case14_ieee | 2,178.08 | 2.1781e+03 | −9e-6 | 0.11% / 0.11% | < 0.1 s |
| case57_ieee | — | 3.7589e+04 | +9e-6 | 0.16% / 0.16% | < 1 s |
| case118_ieee | 97,213.61 | 9.7214e+04 | −4e-6 | 0.90% / 0.91% | 0.3–0.4 s per formulation |
| case300_ieee | 565,219.98 | 5.6522e+05 | −4e-8 | 2.62% / 2.63% | 0.7–0.8 s per formulation |

The full table (up to 3,012 buses, plus the API and SAD stress variants) is in
`energy-grid/README.md`.

## Gotchas (all hit for real)

- **Pyomo can't find Ipopt** unless the env's `bin` is on `PATH` (`No executable found for solver 'ipopt'`).
- **Out-of-service branches** come back from Egret with `pf = None`. Skip them when exporting flows (case300).
- **PyPSA AC power flow after an LOPF diverges** on SciGrid-DE (NaN voltages in most hours) unless all
  generators are set to `control = "PV"`, with the units at one bus set to `PQ`. This is PyPSA's own
  lopf-then-pf recipe. Keep the LOPF dispatch for the generation mix: the PF slack absorbs the losses.
- **Expansion is about 40× the dispatch cost.** SciGrid-DE dispatch solves in 14–17 s; with extendable lines and a
  candidate battery at every bus it takes about 13 min with HiGHS dual simplex on one thread. Race simplex against IPM,
  and limit candidate storage to plausible buses.
- **One representative day undervalues storage.** With annualised costs scaled to 24 h, batteries at
  €120k/MW/yr are not built in the base case. Use multi-day or clustered periods before drawing storage conclusions.
- **MUMPS vs HSL:** PGLib's timings use HSL MA27, which is 2–6× faster than MUMPS. HSL has a separate licence, so compare our times with that in mind.

## Quantum, honestly

- **Today:** unit commitment and network expansion fit QUBO/QAOA only at toy size. The LP and MIP
  solvers above win by orders of magnitude, and the SOC relaxation gives certificates that no
  sampler does.
- **Later:** the optimisation outlook is as for optimization-stack (Grover-type speedups need fault tolerance and
  are eaten by overheads at practical sizes). Power-system dynamics simulation is not a near-term
  quantum target either.
