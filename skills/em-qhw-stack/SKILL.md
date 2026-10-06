---
name: em-qhw-stack
description: Design and simulate superconducting quantum hardware (transmons, resonators, CPW, filters) with open-source electromagnetics on qBraid, including surface-loss (TLS) participation and a predicted T1. AWS Palace for 3D finite-element eigenmode, driven and electrostatic runs, energy-participation quantization into a circuit Hamiltonian, numerical simulation of that Hamiltonian, and comparison with real IBM device parameters. Use when asked to design a qubit or resonator, compute mode frequencies, Q or participation ratios, replace HFSS or Sonnet with open tools, or turn a chip layout into a Hamiltonian.
metadata:
  version: "0.3.0"
  layer: "tool"
  status: "draft"
  verified: "2026-10-01"
---

# Electromagnetics for quantum hardware (em-qhw-stack)

A closed loop on open source (Apache/BSD/GPL, no HFSS or Ansys licence): chip layout, 3D EM eigenmode solve, circuit Hamiltonian, simulation, and comparison with real QPU parameters.

Start from the user's goal: if it is unclear whether they want the best solution or the best quantum solution, ask (see solution-router).

## Get the scripts

The worked example lives in `qubit-design/` of https://github.com/qBraid/open-simulation-examples:

```bash
git clone --depth 1 https://github.com/qBraid/open-simulation-examples
```

All script paths below are relative to that repo. Read `qubit-design/README.md` before improvising.

## 1. Getting Palace onto qBraid

No qBraid environment exists for Palace yet. Check `qbraid envs available` for one first. Otherwise build it as below, then package it with the manage-environments skill for reuse.

- **No binary exists anywhere we checked:** not on conda-forge or PyPI, and no release assets. The routes are Spack (`spack install palace`) or the CMake superbuild, which fetches METIS, hypre, SuperLU_DIST, PETSc/SLEPc, MFEM, libCEED and GSLIB itself. We use the superbuild with conda-forge compilers, OpenMPI 5 and OpenBLAS: `qubit-design/build_palace.sh` and `qubit-design/environment.yml`.
- **Build on local disk (`/tmp`), never in `$HOME`.** The env reaches about 5 GB and the build tree about 2 GB; the home disk is small. `/tmp` is wiped on restart, so package the result before relying on it across instances.
- **Build time:** about 24 min with `-j3`, peak anonymous memory under 6 GB; about 5–8 min with `-j16` on `cpu-32v-128g` ($0.16–0.26).
- **Gotcha: make jobserver mismatch.** conda-forge `make` 4.4 uses a fifo jobserver; the GSLIB sub-build calls the system `gmake` 4.3, which fails with `invalid --jobserver-auth string 'fifo:...'`. Fix: `make -j N --jobserver-style=pipe`.
- **Gotcha: MPI compiler wrappers.** Export `CC=mpicc CXX=mpicxx FC=mpifort` and point `OMPI_CC/OMPI_CXX/OMPI_FC` at the conda compilers. Otherwise the dependency builds pick up the system gcc and link against a mismatched libstdc++.
- **Relocatability (open issue for packaging):** `palace-x86_64.bin` carries an absolute RPATH to the build prefix. Rewrite it to `$ORIGIN/../lib` with `patchelf --set-rpath` before packaging. The env must also ship its own OpenMPI (the `palace -np N` wrapper calls `mpirun` from `PATH`).
- **Fast Palace on a fresh instance:** copy the `ldd` closure of an already-built `palace-x86_64.bin` plus OpenMPI's plugin directories (79 MB compressed) to the same absolute prefix. This skips the superbuild on any instance with the same image. Use `tar -C <prefix>` with relative paths; `./` segments in library paths break naive tarballs.

## 2. Decision rules

| Question | Use |
|---|---|
| Mode frequencies, Q, junction participation of a qubit/resonator chip | **Palace `Eigenmode`**, junction as a `LumpedPort` with `L` (and `C`) |
| S-parameters, filter response, readout-line transmission | **Palace `Driven`** (adaptive fast frequency sweep) |
| Capacitance matrix only (LOM route), surface participation | **Palace `Electrostatic`** |
| Photonics, gratings, broadband time-domain | **Meep** (FDTD) |
| RF antennas, PCB-scale structures | **openEMS** (FDTD) |
| Layout generation | Qiskit Metal, KQCircuits (IQM), DeviceLayout.jl (AWS; used for the full-chip runs here), gdsfactory |

- **Element order:** `Order: 2` for results you report. Order 1 is about 30× faster but noticeably off, so use it only for debugging.
- **Mesh:** refine near the junction, the claw and CPW gaps. AMR (`Model.Refinement.MaxIts > 0`) is the principled route; run it on an instance.
- **Solver:** the default `Linear.Type: Default` (AMS/multigrid with GMRES) is fine. One eigenmode run is dozens of linear solves: CPU-bound and MPI-parallel, so use more ranks, not threads.
- **Losses:** the total Q includes dielectric loss from the configured `LossTan`. A pessimistic loss tangent makes the "T1" look terrible (Palace's example gives 0.72 µs by design). Report the Purcell-only bound (`port-Q.csv`) separately.

## 3. Validate before interpreting

Run Palace's own transmon first: `bash build_palace.sh` (`PREFIX`, `BUILD`, `JOBS` overridable), then `palace -np 3 transmon_coarse.json` (Palace's `examples/transmon`, unchanged except ParaView output on), then `python validate.py`. It compares `eig.csv`, `port-EPR.csv` and `port-Q.csv` with Palace's regression reference (`test/data/regression/ref/transmon/transmon_coarse`). Frequencies must match to at least 1e-4 relative; if they don't, stop. Verified 2026-09-30: 8/8 checks pass (f to 9e-10 / 2e-7), eigenmode solve 39.8 min on 3 ranks, peak 3.4 GB.

## 4. From design to Hamiltonian to simulation

- `derive_hamiltonian.py` implements energy-participation quantization (Minev et al. 2021): E_J from the junction `L`, phi_zpf^2 = p h f / 2E_J, first-order chi and alpha, and **numerical diagonalization** of `sum h f a+a - E_J(cos phi - 1 + phi^2/2)`. Report the numerical values; the first-order ones are a sanity check and differ by about 10% at E_J/E_C of about 50.
- `simulate_dynamics.py` covers truncation convergence, a Rabi trace with leakage to |2>, a Ramsey trace and the dispersive readout lineshapes (chi against kappa).
- `fetch_device_snapshots.py` gives real IBM transmon parameters. `qbraid devices get` and the SDK's `device.metadata()` do not expose T1/T2/frequency today, so the recipe uses the dated calibration snapshots bundled in `qiskit-ibm-runtime` fake backends. Heron (Fez, Marrakesh) snapshots have T1/T2 but no frequency or anharmonicity; Eagle (Sherbrooke, Kyiv) have all four. Label them as snapshots, never as live.

Comparable with devices: frequency, anharmonicity and E_J/E_C. Not comparable: T1. A simulated T1 is a loss-model bound under assumed loss tangents; a device T1 reflects fabrication, TLS defects and packaging.

## 5. Surface loss and predicted T1

A T1 from simulation is a **prediction of the geometry-limited lifetime under assumed loss tangents**, never a measurement. Say so every time.

**Method: two-step surface participation** (Wang et al., APL 107, 162601, 2015):
- **Global:** Palace `Electrostatic` on the layout. Metal as zero-thickness sheets, one terminal per conductor, the qubit mode as the island at 1 V with everything else grounded (grounded transmon) or ±q charges (floating pair). Use a half model across the mirror plane: the natural (Neumann) boundary is exact for symmetric modes and halves time and memory.
- **Local:** `loss/edge2d.py`, a 2D slot model with finite film thickness graded down to 1 nm. The field is read on the midline of the t = 3 nm interface layer, which regularises the corner singularity. Tables S_i(g) converge to 0.03%.
- **Join:** near each edge |E| = K/√u; take K from the surface charge in the band x0/4 < u < x0 (x0 = 4 µm), scale by S_i at the local gap g, and integrate the coarse 3D field directly beyond x0.
- **Conventions:** t = 3 nm and ε = 10 for MA, MS and SA, as in Wang 2015 and Ganjam 2024, so their loss tangents can be reused directly.

```bash
python loss/pipeline.py table 0.2 10.34 4.0 table.json            # 2D edge integrals, about 2 min
python loss/pipeline.py point runs/final '{"W":160,"G":100,"L":310}' table.json --fieldmap
python loss/export_gds.py '{"W":160,"G":100,"L":310}' final_qubit.gds  # analysed geometry as a mask
```

One electrostatic point is 420–460k tets at order 2, about 5 min on 2 ranks, with an 8.3–8.5 GB peak (mostly the always-on error estimator plus ParaView output; `EstimatorMaxIts` barely helps). **Never run Palace on the shared subscription pod:** overlapping runs there were OOM-killed, taking `/tmp` and every agent's work with them. Run it on an instance and copy the small outputs (CSV, participation JSON) back after every point.

**Gotchas:**
- **Palace terminal excitations are not 1 V.** The voltage is in `terminal-V.csv` (we saw 19.41 V). Normalise the fields by it, or every participation is off by V².
- **VTK probe tolerance.** `vtkProbeFilter` snaps points to a neighbouring cell, so a probe 20 nm below a sheet reads the field above it. Use `ComputeToleranceOff()` and `SetTolerance(1e-9)`.
- **gmsh volume classification.** Fragmenting an enclosure around the chip leaves a hollow air volume whose centre of mass lies inside the chip. Classify volumes by bounding box, not centre of mass.
- **Always close the loop with a full-chip eigenmode.** The capacitance (LOM) model put f01 8% high and alpha about 20% too strong against the 3D solve of the same geometry (claw and meander loading, anisotropic sapphire with eps_z = 11.5). Two corrections closed it: re-size the island and resonator from the 3D/LOM E_C ratio, and set L_J from f01 ∝ √E_J on the same mesh. DeviceLayout.jl 1.8.0 (`eigen/gen.jl`) generates the chip, the Palace config and a full-chip GDS in about 1 min after a 3-min precompile; each eigenmode iteration is 5–17 min on 10 ranks of a `cpu-32v-128g`.
- **Validation gap.** Wang 2015 Design A (Table S1) is the reference. Our pad p_MS is about 2× theirs, so raw T1 predictions are conservative; the "calibrated" numbers rescale to Wang's pipeline. Report both. Participation pipelines from different groups differ by O(1), and a loss tangent is only meaningful with the pipeline that fitted it.

**Verified design (2026-10-01):** island 240 × 146 µm in a 150 µm trench, Ta on sapphire, full chip in 3D: f01 4.809 GHz, alpha -270 MHz, E_J/E_C 49, resonator 6.84 GHz, 2chi 1.06 MHz, chi/kappa 0.69. Surface participation 5.2e-4 (Palace's example: 8.5e-4). Predicted T1 142–250 µs (raw to calibrated) against IBM medians of 145–287 µs. Planar edge participation, not the solver, keeps this below the tantalum records (above 0.5 ms).

## 6. What needs an instance

Pick a profile from `qbraid compute list`, launch it through the qbraid-cloud-orchestration skill with `--auto-stop`, copy results back, and terminate it when done.

| Work | Profile | Expect |
|---|---|---|
| Electrostatic points, full-chip eigenmode | `cpu-32v-128g` ($1.92/h) | 5–17 min per eigenmode iteration on 10 ranks |
| AMR run or a finer mesh, 32 ranks | `cpu-32v-128g` | 20–40 min, under $1.50 |
| GPU Palace (`PALACE_WITH_CUDA=ON`, optionally cuDSS) | `gpu-a100-sxm` | build plus run about 1 h, about $2.50; check the driver first (qbraid-cloud-orchestration) |
| Design sweep (claw length against chi, pad gap against alpha) | one run per instance, fanned out | about $5 for 16 designs, then a surrogate |

## Quantum: the short answer

**Runnable test today.** Here quantum is the customer: this stack designs the qubit. The hardware experiment is to characterize real qubits on a QPU and compare them with the design's predicted ranges, for example T1 by delay-sweep circuits (and frequency by spectroscopy where the provider exposes it) on IBM Heron, which needs the user's IBM token via vault-credentials. Treat device data as dated snapshots, and remember a simulated T1 is a bound under assumed loss tangents. Price any QPU run with `qbraid devices get` and confirm with the user before submitting.

For the full answer and the experiment protocol, use the **quantum-readiness** skill (section "Quantum hardware design").

## Verification stamp

- 2026-09-30: Palace v0.18.1 superbuild (conda-forge GCC 15.3, OpenMPI 5, OpenBLAS) on the subscription pod; build 24 min (`-j3`), reference transmon 8/8 checks, $0.
- 2026-10-01: surface-loss pipeline (electrostatics, 14 sweep points plus finals, 2–3 min each on 2–8 ranks; scikit-fem 11 edge model) and three full-chip eigenmode iterations on 10 ranks of a `cpu-32v-128g` (DeviceLayout.jl 1.8.0, Julia 1.12).
- About 1.2 core-hours in total, roughly $1 of instance time. Full results: `qubit-design/README.md`.
