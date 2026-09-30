---
name: cfd-stack
description: Run computational fluid dynamics on qBraid with open-source solvers (OpenFOAM v2412, gmsh, pyvista; SU2 and XLB by decision rule). Use when a user wants airflow, drag/lift, pressure or wake results for a body or duct, a mesh-convergence or validation study, a CFD parameter sweep, or a three.js flow visualisation. Covers the verified install recipe and its traps, solver and machine choice by mesh size, the mandatory validation case, and hand-off to cloud orchestration for sweeps.
metadata:
  version: "0.1.0"
  status: draft
  verified: "2026-09-30"
---

# CFD stack on qBraid

Open-source CFD that stands in for Fluent or STAR-CCM+ on typical external
aerodynamics. The solvers are mature. What an agent has to supply is setup,
meshing, validation and post-processing. This skill covers those parts; the
reference implementation is `wind-tunnel/` in `qBraid/open-simulation-examples`.

## 1. Install (verified 2026-09-30)

```bash
export MAMBA_ROOT_PREFIX=/tmp/mamba          # overlay: fast, large, reset on restart
micromamba create -p /tmp/envs/cfd -f wind-tunnel/env/environment.yml   # ~30 s, 3.4 GB
micromamba run -p /tmp/envs/cfd pimpleFoam -help
```

Keep the env off `$HOME` (it is 3.4 GB and home is small). For reuse across
instances, package it as a qBraid environment (see the environment packaging
skill) rather than rebuilding it.

Traps in `conda-forge::openfoam=2412`. None of them announce themselves:

| Symptom | Cause | Fix |
|---|---|---|
| `mpirun ... -parallel` dies: "The dummy Pstream library cannot be used in parallel mode" | Binaries have an RPATH to `lib/dummy/libPstream.so`; `LD_LIBRARY_PATH` cannot override an RPATH | `mpirun -np N -genv LD_PRELOAD $CONDA_PREFIX/lib/mpich-3.3/libPstream.so <solver> -parallel` |
| snappyHexMesh: `Cannot open etc file "caseDicts/mesh/generation/meshQualityDict"` | The package ships no `etc/caseDicts` | Inline `meshQualityControls` (see `wind-tunnel/ahmed/case/system/snappyHexMeshDict`) |
| `import gmsh` fails | `gmsh` is only the binary | Also install `python-gmsh` at the same version |
| A running `run.sh` dies with a syntax error after you edit it | bash reads scripts incrementally | Never edit a script a live run is executing |

On the pod, `memory.current` counts page cache. Judge headroom from `anon` in
`/sys/fs/cgroup/memory.stat`, and cap MPI ranks at what the pod's `cpu.max`
allows while other agents are running.

## 2. Decision rules

| Situation | Use |
|---|---|
| Incompressible external or internal flow, industrial geometry (STL/CAD) | OpenFOAM: `snappyHexMesh` + `simpleFoam` (steady RANS, k-ω SST) or `pimpleFoam` (transient) |
| Compressible aero, or gradient-based shape optimisation (adjoint) | SU2 (`conda-forge::su2`) |
| Transient, highly resolved, GPU available, simple geometry | XLB (lattice Boltzmann, JAX/Warp) on a GPU instance |
| Many design points, interactive exploration | Sweep with OpenFOAM, then train a surrogate (PhysicsNeMo) on the sweep |

Machine by mesh size (OpenFOAM, about 1 GB RAM per million cells for RANS):

| Cells | Where | Measured or expected |
|---|---|---|
| < 50k (2D) | subscription pod, 1–2 ranks | 27k-cell transient cylinder, 200 D/U: 24 min on 2 ranks (measured) |
| 0.3–1M (3D RANS) | pod, 3 ranks | 592k-cell Ahmed body: snappy 2.7 min serial, 800 SIMPLE iterations 21.7 min on 3 ranks (measured) |
| 1–10M | `cpu-32v-128g` or `cpu-64v-256g` | about 20–30k cells per core for good scaling |
| > 10M, or a sweep | several `cpu-64v-256g` instances via cloud orchestration | one case per instance, and terminate each when done |

## 3. Validation is mandatory

Run a known-answer case before any user case, and report it next to the
result. The reference case is `wind-tunnel/cylinder2d`, a laminar cylinder at
Re = 100, with two meshes:

- Report St (from Cl zero-crossings) and cycle-averaged Cd for coarse and fine
  meshes. Unconfined reference: St 0.164–0.166, Cd 1.33–1.35.
- Verified here, fine mesh with side walls at ±30D: St 0.1685, Cd 1.381, which
  is 1.5–2.7% and 2.3–3.8% above the band. Walls at ±10D gave St 0.1696 and
  Cd 1.391. State offsets like these; don't round them away.
- If a result is off, test the domain (blockage) before the mesh. Side walls
  at ±10D (5% blockage) raise St and Cd by a few percent.

For 3D, compare against a published experiment of the same class (Ahmed body:
Cd 0.257 at 35° slant, Ahmed et al. 1984) and state the gap and the likely
reasons (RANS, wall functions, coarse mesh, no stilts). Never present CFD
numbers without the validation line.

## 4. Show it

Extract a compact payload (surface with Cp, a few hundred streamlines, a
symmetry-plane slice) with pyvista, inline it into a three.js viewer, and
render that in the Agent Canvas: `wind-tunnel/ahmed/extract.py`,
`wind-tunnel/build_viewer.py`. Keep it to a few MB. For large 3D results, serve
ParaView through trame on the instance and expose it through the Lab proxy
(cloud orchestration skill).

## 5. Scale out

For sweeps (speed, angle, geometry parameter), follow the
`qbraid-cloud-orchestration` skill:
- Estimate first: cases × hours × profile rate. For example, 20 cases × 2 h on
  `cpu-64v-256g` at $3.84/h comes to about $154.
- Set `--auto-stop` on every instance.
- Copy results back before terminating.
- Report the cost next to the result.

## Verification stamp

Verified 2026-09-30 on the qBraid subscription pod (`8vCPU_25GB`, shared, at most
3 ranks), with conda-forge `openfoam=2412`, `gmsh`/`python-gmsh` 4.15.2,
`pyvista` 0.49.0 and `mpich` 4.3.2. On-demand cost: $0. The Ahmed body gave
Cd 0.2574 against 0.257 in the experiment; that agreement is partly fortuitous
for coarse RANS. Re-verify on every OpenFOAM or conda-forge package bump.
