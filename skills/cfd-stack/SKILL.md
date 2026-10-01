---
name: cfd-stack
description: Run computational fluid dynamics on qBraid with open-source solvers (OpenFOAM v2412, gmsh, pyvista; SU2 and XLB by decision rule). Use when a user wants airflow, drag/lift, pressure or wake results for a body or duct, a mesh-convergence or validation study, a CFD parameter sweep, or a three.js flow visualisation. Covers the verified install recipe and its traps, solver and machine choice by mesh size, the mandatory validation case, and hand-off to cloud orchestration for sweeps.
metadata:
  version: "0.2.0"
  status: draft
  verified: "2026-10-01"
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
| A second `mpirun` on the same instance segfaults at startup in UCX (`ucs_mpool_grow`) | qBraid instances have a 64 MB `/dev/shm`, already used up by the first job's ranks | `UCX_TLS=self,tcp mpirun ...` for the second job |
| A finer mesh needs more iterations | 3.2M cells was still drifting 1.6% per 150 iterations at iteration 871, where 1.37M had settled by about 700 | Stop on a drift criterion (mean of the last 150 vs the previous 150 under 0.15%), not on a fixed count, and copy forces back after every leg |

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
| 1–1.5M (3D RANS) | a `gpu-l4` box's CPU (cgroup quota 5.1 CPUs), 2 ranks | 1.37M-cell Ahmed body: 1000 SIMPLE iterations about 38 min on 2 ranks (measured 2026-10-01) |
| 1–10M | `cpu-32v-128g` or `cpu-64v-256g` | 3.2M-cell Ahmed body on 10 ranks of a shared `cpu-32v-128g`: about 16 SIMPLE iterations per minute, so 1300 iterations takes about 80 min (measured 2026-10-01). Memory-bandwidth bound: ranks at 50–90% CPU |
| > 10M, or a sweep | several `cpu-64v-256g` instances via cloud orchestration | one case per instance, and terminate each when done |

## 3. Validation is mandatory

Run a known-answer case before any user case, and report it next to the
result. The reference case is `wind-tunnel/cylinder2d`, a laminar cylinder at
Re = 100, with two meshes:

- Report St (from Cl zero-crossings) and cycle-averaged Cd. Unconfined
  reference: St 0.164–0.166, Cd 1.33–1.35.
- **Verified 2026-10-01: inside both bands.** Inlet 20D, side walls ±30D,
  35k cells gives St 0.1652–0.1655 and Cd 1.341–1.346, over 6–7 cycles.
  linearUpwind and central differencing agree within 0.4%.
- **The domain matters more than the mesh.** With the inlet at 10D, the same
  mesh sat 1.8% high on St and 2.5% high on Cd. Side walls at ±10D (5%
  blockage) add another ~1%. When a validation number is off, test the inlet
  distance and blockage before refining.
- When a run is split into legs, the first leg must write a time directory at
  its end (`writeInterval` = the leg's end time). Otherwise
  `startFrom latestTime` silently restarts the next leg from t = 0.

For 3D, compare against a published experiment of the same class, run a mesh
ladder, and state both the gap and whether the flow topology is right. Ahmed
body, verified 2026-10-01, k-ω SST, wall functions, three levels at
0.26/0.59/1.37M cells (half model):

| Slant | Cd coarse / medium / fine | Experiment (Ahmed 1984) | Verdict |
|---|---|---|---|
| 35° | 0.276 / 0.264 / 0.258 | 0.257 | fine within 0.4%, ladder still moving 2.2% per level: close (a fourth level would settle it) |
| 25° | 0.270 / 0.264 / 0.259 | 0.285 | 9% low and still falling about 2% per level; the slant flow stays attached where the experiment separates and reattaches |

The 25° slant is the textbook RANS failure. Say so instead of tuning toward
the number. The honest fix is hybrid RANS/LES (DDES or IDDES) on 20–40M cells
with y+ ≈ 1: about 1–2 days on `cpu-64v-256g`, $90–180 per case. Check the
topology with a near-wall velocity probe along the slant, not just with Cd.
Never present CFD numbers without the validation line.

## 4. Show it

`wind-tunnel/ahmed/extract_v2.py` extracts a compact payload with pyvista:
- the Cp surface, mirrored and decimated to about 48k triangles;
- Q-criterion isosurfaces at Q* = 2 and 8, coloured by streamwise vorticity;
- a 128 × 40 × 42 velocity grid that particles are advected through in the
  browser;
- the symmetry-plane mesh as a PNG with real cell edges.

`wind-tunnel/build_viewer.py` then inlines everything with gzip and base64,
and the browser decodes it with `DecompressionStream`. Two slants, three mesh
levels and a 48-frame cylinder animation fit in about 5 MB. For large 3D results, serve
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

- **2026-10-01 (v2):**
  - Software: conda-forge `openfoam=2412`, `gmsh`/`python-gmsh` 4.15.2,
    `pyvista` 0.49.0, `mpich` 4.3.2.
  - Machines: the qBraid subscription pod (serial, 1 core) and a shared
    `gpu-l4` box. On that box, run CPU jobs on 2 MPI ranks: its cgroup quota
    is 5.1 CPUs even though `nproc` shows 48.
  - Run times: 1.37M-cell SIMPLE, 1000 iterations, about 38 min on 2 ranks.
    0.59M cells about 49 min serial. 35k-cell transient cylinder, t = 0–120,
    17 min on 2 ranks.
  - On-demand cost: $0. This example's share of the pool box was about
    3 CPU-slot-hours.
- **2026-09-30 (v1):** the subscription pod. Results superseded by v2.
- Re-verify on every OpenFOAM or conda-forge package bump.
