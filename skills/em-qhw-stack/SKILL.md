---
name: em-qhw-stack
description: Design and simulate superconducting quantum hardware (transmons, resonators, CPW, filters) with open-source electromagnetics on qBraid, including surface-loss (TLS) participation and predicted T1. AWS Palace for 3D finite-element eigenmode/driven runs, energy-participation quantization into a circuit Hamiltonian, and numerical simulation of that Hamiltonian, set against real IBM device parameters. Use when asked to design a qubit or resonator, compute mode frequencies, Q or participation ratios, replace HFSS or Sonnet with open tools, or turn a chip layout into a Hamiltonian.
metadata:
  version: "0.2.0-draft"
  layer: "1"
  status: "draft - verified on the subscription pod only"
---

# Electromagnetics for quantum hardware (em-qhw-stack)

This is the closed loop that only qBraid can offer in one place. It runs from
the chip layout, through a 3D EM eigenmode solve, to a circuit Hamiltonian, a
simulation, and a comparison with real QPU parameters. Every step here uses
open source (Apache/BSD/GPL). No HFSS or Ansys licence is involved.

The worked, verified example lives in `qBraid/open-simulation-examples`,
`qubit-design/`. Read its README before improvising.

## 1. Operational truth: getting Palace onto qBraid

- **No binary exists anywhere we checked:** it is not on conda-forge or PyPI,
  and has no release assets. The routes are Spack (`spack install palace`) or
  the CMake superbuild, which fetches METIS, hypre, SuperLU_DIST,
  PETSc/SLEPc, MFEM, libCEED and GSLIB itself. We use the superbuild with
  conda-forge compilers, OpenMPI 5 and OpenBLAS: `qubit-design/build_palace.sh`
  and `qubit-design/environment.yml`.
- **Build the toolchain env on the overlay** (`/tmp`), never in `$HOME`. The
  env reaches about 5 GB, the build tree about 2 GB, and the home disk is small.
  `/tmp` is wiped on restart, so package the result as a qBraid environment
  (the `qbraid_3ob4vi` packaging skill) before relying on it across instances.
- **Gotcha: make jobserver mismatch.** The conda-forge `make` 4.4 uses a fifo
  jobserver. The GSLIB sub-build calls the system `gmake` 4.3, which fails with
  `invalid --jobserver-auth string 'fifo:...'`. Fix: `make -j N --jobserver-style=pipe`.
- **Gotcha: MPI compiler wrappers.** Export `CC=mpicc CXX=mpicxx FC=mpifort`
  and point `OMPI_CC/OMPI_CXX/OMPI_FC` at the conda compilers. Otherwise the
  dependency builds pick up the system gcc and link against mismatched libstdc++.
- **Relocatability (open issue for packaging):** the installed
  `palace-x86_64.bin` carries an absolute RPATH to the build prefix. Before
  shipping it as a qBraid env, rewrite it to `$ORIGIN/../lib` with `patchelf`.
  The env also has to ship its own OpenMPI (the wrapper `palace -np N` calls
  `mpirun` from `PATH`).
- **Measured cost:** a build with `-j3` on the shared subscription pod took
  about 24 min, with peak anonymous memory under 6 GB. On `cpu-32v-128g` with
  `-j16` expect roughly 5-8 min ($0.16-0.26).

## 2. Decision rules

| Question | Use |
|---|---|
| Mode frequencies, Q, junction participation of a qubit/resonator chip | **Palace, `Eigenmode`**, junction as a `LumpedPort` with `L` (and `C`) |
| S-parameters, filter response, readout-line transmission | **Palace, `Driven`** (adaptive fast frequency sweep) |
| Capacitance matrix only (LOM route) | **Palace, `Electrostatic`** |
| Photonics, gratings, broadband time-domain | **Meep** (FDTD) |
| RF antennas, PCB-scale structures | **openEMS** (FDTD) |
| Layout generation | Qiskit Metal, KQCircuits (IQM), DeviceLayout.jl (AWS, used for Palace's transmon mesh), gdsfactory |

- **Element order:** `Order: 2` for results you report. Order 1 is about 30×
  faster but noticeably off (Palace docs), so use it only for debugging.
- **Mesh:** refine near the junction, the claw and CPW gaps. AMR
  (`Model.Refinement.MaxIts > 0`) is the principled route, but it needs a real
  instance, not the pod.
- **Solver:** the default `Linear.Type: Default` (AMS/multigrid with GMRES) is
  fine. One eigenmode run is dozens of linear solves, so it is CPU-bound and
  MPI-parallel. Use more ranks, not threads.
- **Losses:** the total Q includes dielectric loss from the configured
  `LossTan`. A pessimistic loss tangent makes the "T1" look terrible. Report
  the Purcell-only bound (`port-Q.csv`) separately.

## 3. Verified recipe (stamp)

> Verified 2026-09-30: Palace v0.18.1 superbuild, conda-forge GCC + OpenMPI 5 +
> OpenBLAS, qBraid subscription pod (8 vCPU tier, shared with other workloads),
> `palace -np 3 transmon_coarse.json`. Build 24 min (-j3), eigenmode solve 39.8 min, peak 3.4 GB, validation 8/8 against Palace regression reference (f to 9e-10 / 2e-7). $0 (no instance).

1. `bash build_palace.sh`, with `PREFIX`, `BUILD` and `JOBS` overridable.
2. `palace -np 3 transmon_coarse.json`, which is Palace's own
   `examples/transmon`, unchanged except that ParaView output is on.
3. **Validate before interpreting.** `python validate.py` compares `eig.csv`,
   `port-EPR.csv` and `port-Q.csv` against Palace's regression reference
   (`test/data/regression/ref/transmon/transmon_coarse`). Frequencies must
   match to at least 1e-4 relative. If they don't, stop.

## 4. From design to Hamiltonian to simulation

- `derive_hamiltonian.py` implements energy-participation quantization (Minev
  et al. 2021). It gives E_J from the junction `L`, phi_zpf^2 = p h f / 2E_J,
  first-order chi and alpha, and **numerical diagonalization** of
  `sum h f a+a - E_J(cos phi - 1 + phi^2/2)`. Report the numerical values. The
  first-order ones are a sanity check: they differ by about 10% at E_J/E_C of
  about 50.
- `simulate_dynamics.py` covers truncation convergence, a Rabi trace with
  leakage to |2>, a Ramsey trace and the dispersive readout lineshapes
  (chi against kappa).
- `fetch_device_snapshots.py` gives the real IBM transmon parameters.
  **Truth about data sources:** `qbraid devices get` and the SDK's
  `device.metadata()` do not expose T1/T2/frequency today. The recipe uses the
  dated calibration snapshots bundled in `qiskit-ibm-runtime` fake backends.
  Heron (Fez, Marrakesh) snapshots have T1/T2 but no frequency or
  anharmonicity. Eagle (Sherbrooke, Kyiv) have all four. Label them as
  snapshots and never present them as live.

What is comparable: frequency, anharmonicity and E_J/E_C, where the design
lands inside or outside the fleet's range. What is not: T1. The simulated T1
is a loss-model bound set by the assumed loss tangents, while the device T1
reflects fabrication, TLS defects and packaging.

## 6. Surface loss and predicted T1 (v2 method)

A T1 number from a simulation is a **prediction of the geometry-limited
lifetime under assumed loss tangents**, never a measurement. Say so every time.

**Method: two-step surface participation**, after Wang et al., APL 107, 162601 (2015).
- **Global.** Palace `Electrostatic` on the layout. Metal is zero-thickness
  sheets on the substrate, there is one terminal per conductor, and the qubit
  mode is the island at 1 V with everything else grounded (a grounded transmon)
  or ±q charges (a floating pair). Use a half model across the mirror plane:
  the natural (Neumann) boundary is exact for symmetric modes and halves time
  and memory.
- **Local.** The `loss/edge2d.py` 2D slot model has finite film thickness and
  grading down to 1 nm. The field is read on the midline of the t = 3 nm
  interface layer (Wang Fig. S2), which regularises the corner singularity.
  The tables S_i(g) are converged to 0.03%.
- **Join.** Near each edge, |E| = K/√u. K is taken from the surface charge in
  the band x0/4 < u < x0 (x0 = 4 µm) and multiplied by S_i at the local gap g.
  Beyond x0, the coarse 3D field is integrated directly.
- **Conventions:** t = 3 nm and ε = 10 for MA, MS and SA. They are the same
  as Wang 2015 and Ganjam 2024, so their loss tangents can be reused directly.

**Gotchas**
- **Palace terminal excitations are not 1 V.** The terminal voltage is in
  `terminal-V.csv` (we saw 19.41 V). Normalise the fields by it, or every
  participation is off by V².
- **VTK probe tolerance.** `vtkProbeFilter` snaps points within its
  tolerance to the neighbouring cell. A probe 20 nm below a sheet then reads
  the field above it. Use `ComputeToleranceOff()` and `SetTolerance(1e-9)`.
- **Volume classification in gmsh.** Fragmenting an enclosure around the chip
  leaves a hollow air volume whose centre of mass lies inside the chip.
  Classify volumes by bounding box, not by centre of mass.
- **Palace electrostatics memory.** A 420-460k-tet, order-2 run peaks at
  about 8.3-8.5 GB, mostly the always-on error estimator plus ParaView output.
  `EstimatorMaxIts` barely helps. Plan one job at a time on a 25 GB pod.
- **Shipping Palace to another box.** Copy only the `ldd` closure of
  `palace-x86_64.bin` plus OpenMPI's plugin directories (79 MB compressed) to
  the same absolute prefix; the RPATH is absolute. Use `tar -C <prefix>` with
  relative paths, because `./` segments in library paths break naive tarballs.

**Validation, and the honest gap.** Wang 2015 Design A (Table S1) is the
reference. Our pad p_MS is about 2× theirs, so our raw T1 predictions are
conservative. The "calibrated" numbers rescale to Wang's pipeline. Always
report both. Participation pipelines from different groups differ by O(1),
and a loss tangent is only meaningful together with the pipeline that fitted it.

## 7. What needs an instance

- AMR or fine meshes: `cpu-32v-128g` ($1.92/h). For a transmon AMR run expect
  about 20-40 min with 32 ranks, so under $1.50.
- GPU Palace (`PALACE_WITH_CUDA=ON`, optionally cuDSS): A100/H100 build and
  run. Check the driver first (see qbraid-cloud-orchestration).
- A design sweep (claw length against chi, pad gap against alpha): one Palace
  run per instance, fanned out with qbraid-cloud-orchestration, then a
  surrogate.
- Closing the loop on hardware (spectroscopy/T1 on a real QPU) spends money
  and needs the user's IBM token. Estimate and ask first.
