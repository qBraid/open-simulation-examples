---
name: cfd-stack
description: Run computational fluid dynamics on qBraid with open-source solvers (OpenFOAM v2412, gmsh, pyvista; SU2 and XLB by decision rule). Use when a user wants airflow, drag/lift, pressure or wake results for a body or duct, a mesh-convergence or validation study, a CFD parameter sweep, an open alternative to Ansys Fluent or STAR-CCM+, or a three.js flow visualisation. Covers the verified install recipe and its traps, solver and machine choice by mesh size, the mandatory validation case, and hand-off to cloud orchestration for sweeps.
metadata:
  version: "0.3.0"
  status: "provisional"
  verified: "2026-10-01"
---

# CFD stack on qBraid

Open-source CFD that stands in for Fluent or STAR-CCM+ on typical external
aerodynamics. The solvers are mature; what an agent has to supply is setup,
meshing, validation and post-processing.

Start from the user's goal: if it is unclear whether they want the best
solution or the best quantum solution, ask (see solution-router).

## Get the scripts

The worked example lives in `wind-tunnel/` of
https://github.com/qBraid/open-simulation-examples:

```bash
git clone --depth 1 https://github.com/qBraid/open-simulation-examples
```

All script paths below are relative to that repo. Its README has the full
reproduce block and every result.

## 1. Environment (verified 2026-10-01)

No qBraid environment carries this stack yet. Check `qbraid envs available`
for an OpenFOAM environment first. Otherwise build it from the repo's
environment file:

```bash
export MAMBA_ROOT_PREFIX=/tmp/mamba          # local disk: fast, large, reset on restart
micromamba create -p /tmp/envs/cfd -f wind-tunnel/env/environment.yml   # about 1 min, 3.4-4.2 GB
micromamba run -p /tmp/envs/cfd pimpleFoam -help
```

Keep the env off `$HOME`: it is several GB and home is small. For reuse across
instances, package it as a qBraid environment with the manage-environments
skill instead of rebuilding it.

Traps in `conda-forge::openfoam=2412`. None of them announce themselves:

| Symptom | Cause | Fix |
|---|---|---|
| `mpirun ... -parallel` dies: "The dummy Pstream library cannot be used in parallel mode" | Binaries have an RPATH to `lib/dummy/libPstream.so`; `LD_LIBRARY_PATH` cannot override an RPATH | `mpirun -np N -genv LD_PRELOAD $CONDA_PREFIX/lib/mpich-3.3/libPstream.so <solver> -parallel` |
| snappyHexMesh: `Cannot open etc file "caseDicts/mesh/generation/meshQualityDict"` | The package ships no `etc/caseDicts` | Inline `meshQualityControls` (see `wind-tunnel/ahmed/case/system/snappyHexMeshDict`) |
| `import gmsh` fails | `gmsh` is only the binary | Also install `python-gmsh` at the same version |
| A running `run.sh` dies with a syntax error after you edit it | bash reads scripts incrementally | Never edit a script a live run is executing |
| A second `mpirun` on the same instance segfaults at startup in UCX (`ucs_mpool_grow`) | qBraid instances have a 64 MB `/dev/shm`, already used up by the first job's ranks | `UCX_TLS=self,tcp mpirun ...` for the second job |
| A finer mesh needs more iterations | 3.2M cells was still drifting 1.6% per 150 iterations at iteration 871, where 1.37M had settled by about 700 | Stop on a drift criterion (mean of the last 150 vs the previous 150 under 0.15%), not on a fixed count, and copy forces back after every leg |

Size MPI runs from the cgroup quota, not `nproc`: a `gpu-l4` instance's
cgroup allows about 5 CPUs even though `nproc` shows 48. On the subscription
pod, `memory.current` counts page cache; judge headroom from `anon` in
`/sys/fs/cgroup/memory.stat`, and do not run solvers there while other agents
share it.

## 2. Decision rules

| Situation | Use |
|---|---|
| Incompressible external or internal flow, industrial geometry (STL/CAD) | OpenFOAM: `snappyHexMesh` + `simpleFoam` (steady RANS, k-ω SST) or `pimpleFoam` (transient) |
| Compressible aero, or gradient-based shape optimisation (adjoint) | SU2 (`conda-forge::su2`) |
| Transient, highly resolved, GPU available, simple geometry | XLB (lattice Boltzmann, JAX/Warp) on a GPU instance |
| Many design points, interactive exploration | Sweep with OpenFOAM, then train a surrogate (PhysicsNeMo) on the sweep |

Machine by mesh size (OpenFOAM, about 1 GB RAM per million cells for RANS).
Pick a profile from `qbraid compute list`:

| Cells | Where | Measured |
|---|---|---|
| < 50k (2D) | subscription pod or a small CPU instance, 1-2 ranks | 35k-cell transient cylinder, t = 0-120: 17 min on 2 ranks |
| 0.3-1.5M (3D RANS) | `cpu-8v-32g`, 2-4 ranks | 1.37M-cell Ahmed body: 1000 SIMPLE iterations about 38 min on 2 ranks |
| 1-10M | `cpu-32v-128g` or `cpu-64v-256g` | 3.2M-cell Ahmed body on 10 ranks of a `cpu-32v-128g`: about 16 SIMPLE iterations per minute, so 1300 iterations take about 80 min. Memory-bandwidth bound: ranks at 50-90% CPU |
| > 10M, or a sweep | several `cpu-64v-256g` instances via cloud orchestration | one case per instance; terminate each when done |

## 3. Validation is mandatory

Run a known-answer case before any user case, and report it next to the
result. The reference case is `wind-tunnel/cylinder2d`, a laminar cylinder at
Re = 100:

- Report St (from Cl zero-crossings) and cycle-averaged Cd. Unconfined
  reference: St 0.164-0.166, Cd 1.33-1.35.
- **Verified 2026-10-01: inside both bands.** Inlet 20D, side walls ±30D,
  35k cells gives St 0.1652-0.1655 and Cd 1.341-1.346, over 6-7 cycles.
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
body, k-ω SST, wall functions, half model at 0.26/0.59/1.37M cells:

| Slant | Cd coarse / medium / fine | Experiment (Ahmed 1984) | Verdict |
|---|---|---|---|
| 35° | 0.276 / 0.264 / 0.258 | 0.257 | fine within 0.4%, ladder still moving 2.2% per level: close |
| 25° | 0.270 / 0.264 / 0.259 | 0.285 | 9% low and still falling; the slant flow stays attached where the experiment separates and reattaches |

The 25° slant is the textbook RANS failure. Say so instead of tuning toward
the number. The honest fix is hybrid RANS/LES (DDES or IDDES) on 20-40M cells
with y+ ≈ 1: about 1-2 days on `cpu-64v-256g`, $90-180 per case. Check the
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
levels and a 48-frame cylinder animation fit in about 5 MB. Render it with the
agent-canvas skill. For large 3D results, serve ParaView through trame on the
instance and expose it through the Lab proxy (qbraid-cloud-orchestration).

## 5. Scale out

For sweeps (speed, angle, geometry parameter), follow the
qbraid-cloud-orchestration skill:
- Estimate first: cases × hours × profile rate. For example, 20 cases × 2 h on
  `cpu-64v-256g` at $3.84/h comes to about $154. Confirm with the user.
- Set `--auto-stop` on every instance.
- Copy results back before terminating.
- Report the cost next to the result.

## Quantum: the short answer

**No credible near-term path.** Quantum linear-system and quantum
lattice-Boltzmann algorithms exist on paper, but loading a full flow field
into a quantum state and reading it back out cancels the speedup, and the
proposals need fault-tolerant hardware. The best answer today is a validated
classical run. For users who want to learn, a toy linear-solve circuit on a
simulator is fine; any QPU run is priced with `qbraid devices get` and
confirmed with the user first. For the full answer and the experiment
protocol, use the **quantum-readiness** skill (section "Differential
equations and simulation").

## Verification stamp

- 2026-10-01: conda-forge `openfoam=2412`, `gmsh`/`python-gmsh` 4.15.2,
  `pyvista` 0.49.0, `mpich` 4.3.2.
- Machines: the subscription pod (serial), a `gpu-l4` instance's CPUs
  (2 ranks), and a `cpu-32v-128g` (10 ranks) for the 3.2M-cell level.
- Wall time: 1.37M cells, 1000 SIMPLE iterations, about 38 min on 2 ranks;
  cylinder validation 17 min on 2 ranks.
- Cost: under $1 for the validation and three-level ladder; the 3.2M-cell
  attempt about 1.7 h on 10 cores of a `cpu-32v-128g`.
- Re-verify on every OpenFOAM or conda-forge package bump.
