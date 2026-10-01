---
name: fea-stack
description: Structural finite elements and topology optimization on qBraid with open-source tools (SIMP top88/top3d ports, scikit-fem, FEniCSx, CHOLMOD). Use when a user wants to size, verify or lightweight a part, reproduce a published topology-optimization benchmark, design a bracket or beam for given loads, or export a printable STL, as an open alternative to Abaqus/ATOM, Ansys or OptiStruct for linear-elastic design.
---

# FEA stack: verified structural analysis and topology optimization

Draft. It was distilled from the `topology-optimization/` example in
qBraid/open-simulation-examples, and every number below comes from a run in that
example.

## What to reach for

| Need | Tool | Why |
|---|---|---|
| Compliance topology optimization, 2D | `topopt.top88` (port of Andreassen et al. 2011) | Matches the published code to round-off; seconds per run |
| Compliance topology optimization, 3D | `topopt.simp3d` / `topopt.top3d` (port of Liu & Tovar 2014) | Same H8 element, filter and OC update as the reference; adds passive regions, multiple load cases and Heaviside projection |
| Independent FE check, any geometry | scikit-fem with P2 tetrahedra | Pure Python, higher order, converges fast |
| Large or nonlinear FE, PETSc-scale | FEniCSx (dolfinx, conda-forge) | Weak forms in UFL, MPI/PETSc solvers |
| Linear solves | CHOLMOD via scikit-sparse | 5 to 20 times faster than SuperLU for SPD stiffness matrices |
| Printable output | scikit-image marching cubes, then trimesh Taubin smoothing, then STL | Watertight mesh for slicing |

## Operational truth on qBraid (verified 2026-10-01)

- **Environment.** Use conda-forge: `python=3.12 numpy scipy scikit-sparse scikit-image scikit-fem trimesh matplotlib`, plus `octave` if you need to run the MATLAB reference codes. The full environment is 2.5 GB. Put it on local disk (`/tmp/...`) on GPU boxes whose home is the network filesystem.
- **scikit-sparse 0.5 changed its API.** `cholesky(A)` is no longer callable as a solver. Use `cho_factor(A).solve(b)`. `topopt.solve_spd` handles both versions.
- **conda-forge Octave 10.3 doesn't know its own install prefix.** It fails with `'mkdir' undefined` or `'repmat' undefined`. Fix it with `export OCTAVE_HOME=<env prefix>`.
- **Don't redistribute top88.m or top3d.m.** Their authors reserve the rights. Fetch them at run time (`reference/fetch_reference.sh`) and make headless copies that only remove plotting and save the design.
- **CPU, not GPU.** These problems are sparse-direct bound. Up to about 200k unknowns (a 64 × 32 × 24 brick grid) a factorization takes seconds on 2 threads. 160 design iterations of the bracket take about 15 to 30 minutes.
- **Never edit a shell script while bash is running it.** bash reads scripts incrementally, so the edit corrupts the run. Write a new script instead.

## Decision rules

- **Resolution.** Aim for at least 6 to 10 trilinear bricks across a member. The H8 brick shear-locks in bending: a cantilever with 8 bricks through the depth is 1.0 % too stiff against the 3D continuum, and 3.6 % with 4. Use P2 tetrahedra (scikit-fem or FEniCSx) when you need a stress or deflection verdict, not a layout.
- **Filter radius.** Use 0.04 × domain width for 2D benchmarks (the top88 convention), and 1.5 to 2 bricks in 3D. Too small gives checkerboards and members thinner than the mesh can represent.
- **Crisp designs.** Use density filter plus Heaviside projection with continuation (beta 1 → 16). The grey fraction should be under about 2 % before you export an STL.
- **Stress.** Report the 99th percentile of element von Mises stress, excluding the elements next to clamps and loaded holes (boundary-condition singularities). Quote it against yield as a linear-elastic safety factor and say so.
- **Stopping rule.** top3d's default (`change < 0.01`) may never trigger, because the OC update oscillates around 0.015. The reference code then stops at its 200-iteration cap. When comparing to a reference, compare at matched iteration counts.

## Mandatory verification (do these before claiming a design)

1. **Element check.** The H8 stiffness matrix must match top3d's closed-form `lk_H8` (achieved: max difference 1.7e-16).
2. **Analytic check.** A clamped cantilever (L/h = 10, ν = 0.3) has Timoshenko tip deflection 4030.6/E for a unit load. The 3D continuum answer is 0.69 % below that, because the 3D clamp restrains the section. H8 and P2 must converge to the same continuum value (achieved: they agree within 0.14 %).
3. **Benchmark check.** Reproduce top88 (MBB 60 × 20 / 150 × 50 / 300 × 100, volume fraction 0.5, p = 3, filter radius 0.04 × width; sensitivity and density filters) and top3d (60 × 20 × 4, 0.3, 3, 1.5) against the published codes run in Octave. The bar is a compliance difference under 1 %. Achieved: the ports are identical to about 1e-8 for 50 or more iterations, and the 3D case at 200 iterations differs by 0.003 %. See `results/benchmarks.json`.

## Verified recipe (stamp)

| Run | Machine | Time | Cost |
|---|---|---|---|
| top88 2D benchmarks (Python ports) | pod, 1 thread | 0.4 s to 35 s per case | $0 |
| top3d 60 × 20 × 4, Python port | pod, 1 thread | 72 s (135 iterations), 233 s (forced to 200) | $0 |
| Reference codes in Octave | qBraid L4 pool box, 1 thread | minutes; the filter loop dominates at radius 16 | shared pool |
| Titanium bracket, 49k bricks, 161k unknowns, 3 load cases | pool box, 2 threads | see `results/bracket.json` | shared pool |

## Pointers

- Running the bracket or a design sweep on a bigger CPU box, and watching cost: **qbraid-cloud-orchestration**.
- Showing results with three.js in the Agent Canvas: `topology-optimization/viewer.html` is the pattern. It uses marching cubes on interpolated density snapshots in the browser, plus the Python-smoothed final mesh.
- Benchmark discipline (criteria before results, honest negatives): **qBraid Quantum Scientist**.

## Limits to state every time

Linear elastic, small strain, compliance objective. No contact, plasticity, fatigue, buckling, overhang or minimum-member-size manufacturing constraints. Commercial tools (Abaqus/ATOM, Ansys, OptiStruct) add those constraints and certified solvers. The optimization core (SIMP, filters, projection, multiple load cases) is the same algorithm family.
