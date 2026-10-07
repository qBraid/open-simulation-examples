---
name: fea-stack
description: Structural finite elements and topology optimization on qBraid with open-source tools (SIMP top88/top3d ports, scikit-fem, FEniCSx, CHOLMOD). Use when a user wants to size, verify or lightweight a part, reproduce a published topology-optimization benchmark, design a bracket or beam for given loads, or export a printable STL, as an open alternative to Abaqus/ATOM, Ansys or OptiStruct for linear-elastic design.
metadata:
  version: "0.3.0"
  status: "provisional"
  verified: "2026-10-01"
---

# FEA stack: verified structural analysis and topology optimization

Open-source linear-elastic FE and SIMP topology optimization, checked against
the published reference codes before any design claim.

Start from the user's goal: if it is unclear whether they want the best
solution or the best quantum solution, ask (see solution-router).

## Get the scripts

The worked example lives in `topology-optimization/` of
https://github.com/qBraid/open-simulation-examples:

```bash
git clone --depth 1 https://github.com/qBraid/open-simulation-examples
```

All script paths below are relative to that repo. Its README has the full
reproduce block and every result.

## What to reach for

| Need | Tool | Why |
|---|---|---|
| Compliance topology optimization, 2D | `topopt.top88` (port of Andreassen et al. 2011) | Matches the published code to round-off; seconds per run |
| Compliance topology optimization, 3D | `topopt.simp3d` / `topopt.top3d` (port of Liu & Tovar 2014) | Same H8 element, filter and OC update as the reference; adds passive regions, multiple load cases and Heaviside projection |
| Independent FE check, any geometry | scikit-fem with P2 tetrahedra | Pure Python, higher order, converges fast |
| Large or nonlinear FE, PETSc-scale | FEniCSx (dolfinx, conda-forge) | Weak forms in UFL, MPI/PETSc solvers |
| Linear solves | CHOLMOD via scikit-sparse | 5 to 20 times faster than SuperLU for SPD stiffness matrices |
| Printable output | scikit-image marching cubes, then trimesh Taubin smoothing, then STL | Watertight mesh for slicing |

## Environment and operational truth (verified 2026-10-01)

No qBraid environment carries this stack yet. Check `qbraid envs available`
first. Otherwise build it from conda-forge (about 2.5 GB; put it on local disk
such as `/tmp` on instances whose home is a network filesystem), and package it
with the manage-environments skill for reuse:

```bash
micromamba create -p ./env -c conda-forge python=3.12 numpy scipy scikit-sparse scikit-image scikit-fem trimesh matplotlib octave
export OCTAVE_HOME=$PWD/env            # conda-forge Octave 10.3 needs this
```

- **scikit-sparse 0.5 changed its API.** `cholesky(A)` is no longer callable as a solver. Use `cho_factor(A).solve(b)`. `topopt.solve_spd` handles both versions.
- **conda-forge Octave 10.3 doesn't know its own install prefix.** It fails with `'mkdir' undefined` or `'repmat' undefined`. Fix it with `export OCTAVE_HOME=<env prefix>`. Octave is only needed to run the MATLAB reference codes.
- **Don't redistribute top88.m or top3d.m.** Their authors reserve the rights. Fetch them at run time (`reference/fetch_reference.sh`) and make headless copies that only remove plotting and save the design.
- **CPU, not GPU.** These problems are sparse-direct bound. Up to about 200k unknowns (a 64 × 32 × 24 brick grid) a factorization takes seconds on 2 threads. The 2 mm bracket (160 design iterations) took 21 min on 1 thread; the 1.5 mm version is an estimated 1 to 2 hours on a `cpu-8v-32g` instance.
- **Never edit a shell script while bash is running it.** bash reads scripts incrementally, so the edit corrupts the run. Write a new script instead.

## Decision rules

- **Resolution.** Aim for at least 6 to 10 trilinear bricks across a member. The H8 brick shear-locks in bending: a cantilever with 8 bricks through the depth is 1.0 % too stiff against the 3D continuum, and 3.6 % with 4. Use P2 tetrahedra (scikit-fem or FEniCSx) when you need a stress or deflection verdict, not a layout.
- **Filter radius.** Use 0.04 × domain width for 2D benchmarks (the top88 convention), and 1.5 to 2 bricks in 3D. Too small gives checkerboards and members thinner than the mesh can represent.
- **Crisp designs.** Use density filter plus Heaviside projection with continuation (beta 1 → 16). The grey fraction should be under about 2 % before you export an STL.
- **Stress.** Report the 99th percentile of element von Mises stress, excluding the elements next to clamps and loaded holes (boundary-condition singularities). Quote it against yield as a linear-elastic safety factor and say so.
- **Stopping rule.** top3d's default (`change < 0.01`) may never trigger, because the OC update oscillates around 0.015. The reference code then stops at its 200-iteration cap. When comparing to a reference, compare at matched iteration counts.
- **Bigger runs.** For a finer bracket or a design sweep, launch a CPU instance (`cpu-8v-32g` or `cpu-32v-128g` from `qbraid compute list`) through the qbraid-cloud-orchestration skill, with `--auto-stop`, and terminate it when done.

## Mandatory verification (do these before claiming a design)

1. **Element check.** The H8 stiffness matrix must match top3d's closed-form `lk_H8` (achieved: max difference 1.7e-16).
2. **Analytic check.** A clamped cantilever (L/h = 10, ν = 0.3) has Timoshenko tip deflection 4030.6/E for a unit load. The 3D continuum answer is 0.69 % below that, because the 3D clamp restrains the section. H8 and P2 must converge to the same continuum value (achieved: they agree within 0.14 %). Script: `validate_fe.py`.
3. **Benchmark check.** Reproduce top88 (MBB 60 × 20 / 150 × 50 / 300 × 100, volume fraction 0.5, p = 3, filter radius 0.04 × width; sensitivity and density filters) and top3d (60 × 20 × 4, 0.3, 3, 1.5) against the published codes run in Octave. The bar is a compliance difference under 1 %. Achieved: the ports are identical to about 1e-8 for 50 or more iterations, and the 3D case at 200 iterations differs by 0.003 %. Scripts: `run_benchmarks.py`, `compare.py`; results in `results/benchmarks.json`.

## Show it

`topology-optimization/viewer.html` is the pattern: marching cubes on
interpolated density snapshots in the browser, plus the Python-smoothed final
mesh, von Mises heat map and load glyphs. Build it with
`export_viewer_data.py` and `build_viewer.py`, and render it with the
agent-canvas skill. Benchmark discipline (criteria before results, honest
negatives) follows **qbraid-quantum-scientist**.

## Limits to state every time

Linear elastic, small strain, compliance objective. No contact, plasticity, fatigue, buckling, overhang or minimum-member-size manufacturing constraints. Commercial tools (Abaqus/ATOM, Ansys, OptiStruct) add those constraints and certified solvers. The optimization core (SIMP, filters, projection, multiple load cases) is the same algorithm family.

## Quantum: the short answer

**No credible near-term path.** The FE solve is a large sparse linear system;
quantum linear-system algorithms need fault-tolerant hardware and lose their
speedup to loading the stiffness data and reading out the displacement field.
Topology optimization written as a QUBO has been shown only at toy size, far
below what a SIMP run solves in seconds. Give the validated classical answer;
any QPU run is priced with `qbraid devices get` and confirmed with the user
first. For the full answer and the experiment protocol, use the
**quantum-readiness** skill (section "Differential equations and simulation";
for the QUBO angle, section "Optimization").

## Verification stamp

- 2026-10-01: conda-forge python 3.12, numpy 2.5, scipy 1.18, scikit-sparse 0.5 (CHOLMOD), scikit-fem 12.0, Octave 10.3 for the reference codes.
- Machines: the subscription pod (1 thread) for the Python runs and the 2 mm bracket; a `gpu-l4` instance's CPU (1 thread) for the Octave reference runs.
- Wall time: bracket 21 min; top88 suite 1 h (the 300 × 100 density-filter case alone 52 min); top3d 1.2-3.9 min; reference codes about 50 min.
- Cost: $0 on the pod; about 1 CPU-hour on the instance.
