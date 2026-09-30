# Design a qubit, then run it

This example takes a transmon qubit with a quarter-wave readout resonator from
3D electromagnetic simulation to a circuit Hamiltonian, then to a simulation
of that Hamiltonian, and sets the result beside real IBM transmons. Every step
uses open-source software. It stands in for an Ansys HFSS workflow.

```
chip geometry ──► AWS Palace eigenmode (3D FEM) ──► validate vs. Palace's reference
                                  │
                                  ▼
         energy-participation quantization (E_J, phi_zpf, chi, alpha)
                                  │
                                  ▼
  numerical Hamiltonian ──► Rabi / Ramsey / dispersive readout ──► vs. IBM device parameters
                                  │
                                  ▼
                   viewer.html (three.js: chip + mode fields + panels)
```

Open `viewer.html` in a browser, or in the qBraid Agent Canvas with
`qbraid-canvas viewer.html`. It is a single 0.5 MB file, and three.js loads
from jsDelivr.

## Results

> **Verified 2026-09-30.** Palace v0.18.1 (CMake superbuild, conda-forge GCC 15.3,
> Open MPI 5.0.11, OpenBLAS) on the qBraid subscription pod (8 vCPU tier,
> shared with other workloads), 3 MPI ranks. Build 24 min, eigenmode solve
> 39.8 min, peak memory 3.4 GB. Cost $0 (no on-demand instance).

### 1. Validation (mandatory before interpreting anything)

The run reproduces Palace's own CI regression reference for this example
(`test/data/regression/ref/transmon/transmon_coarse`) on all 8 quantities:

| quantity | this run | reference | rel. diff |
|---|---|---|---|
| qubit mode f | 4.0991155 GHz | 4.0991155 GHz | 8.9e-10 |
| qubit mode Q | 18553 | 18553 | 1.3e-09 |
| qubit junction EPR p | 0.99191 | 0.99191 | 7.3e-10 |
| qubit Q_ext (port 1) | 3.262e7 | 3.262e7 | 1.9e-06 |
| resonator mode f | 5.6032672 GHz | 5.6032660 GHz | 2.2e-07 |
| resonator mode Q | 7911.2 | 7911.2 | 1.1e-06 |
| resonator junction EPR p | 0.0014837 | 0.0014837 | 4.7e-07 |
| resonator Q_ext (port 1) | 2.792e4 | 2.792e4 | 2.4e-06 |

Tolerances are 1e-4 for frequencies and 2e-2 for Q, EPR and Q_ext
(`validate.py`). Full data is in `results/validation.json`.

### 2. From design to Hamiltonian (`derive_hamiltonian.py`)

This uses energy-participation quantization (Minev et al., *npj Quantum Inf.*
7, 131, 2021). The junction is the lumped `L = 14.86 nH` port in the Palace
config, so E_J/h = 11.00 GHz. The truncated Hamiltonian
`sum h f_m a+a - E_J (cos phi - 1 + phi^2/2)` is diagonalized with 12 qubit
levels and 14 resonator levels. It is converged: going from 8x10 to 12x14
changes alpha by 1.5 MHz.

| parameter | numerical | first-order EPR |
|---|---|---|
| qubit f01 | **3.901 GHz** (linear mode 4.099 GHz) | 3.911 GHz |
| anharmonicity alpha | **-212.9 MHz** | -187.9 MHz |
| resonator | **5.603 GHz** | |
| dispersive shift (2 chi, resonator pull g→e) | **-601 kHz** | -768 kHz |
| E_J / E_C | **51.7** (transmon regime) | |
| resonator kappa/2pi (both feedline ports) | 410 kHz | |
| chi / kappa | 1.46 | |
| qubit T1, loss-limited | 0.72 us (see limits) | |
| qubit T1, Purcell only | 623 us | |

The first-order and numerical values differ by about 12% on alpha and 22% on
chi. That is the expected size of higher-order corrections at E_J/E_C of
about 50, and it is why the numerical value is the one to report.

### 3. From Hamiltonian to simulation (`simulate_dynamics.py`)

- **Rabi, 25 MHz drive, 3-level transmon with T1:** peak leakage to |2> is
  0.74%. A plain (non-DRAG) pulse at this anharmonicity leaks measurably, and
  that is what DRAG corrects.
- **Dispersive readout:** the resonator line moves 601 kHz when the qubit is
  excited, against a 708 kHz loaded linewidth, so the two states are resolvable.
- **Ramsey** at 5 MHz detuning, with T2 = 2 T1 assumed (no pure dephasing
  modelled).

### 4. Beside real IBM transmons (`fetch_device_snapshots.py`)

| | f01 (GHz) | alpha (MHz) | T1 (us) | T2 (us) |
|---|---|---|---|---|
| **this design** | 3.901 | -213 | 0.72* | – |
| Sherbrooke (Eagle), median | 4.794 | -311 | 278 | 170 |
| Kyiv (Eagle), median | 4.613 | -311 | 287 | 118 |
| Fez (Heron), median | n/p | n/p | 145 | 88 |
| Marrakesh (Heron), median | n/p | n/p | 197 | 118 |

**What is comparable:** frequency and anharmonicity. The design sits about
0.7 GHz lower and has a weaker anharmonicity (about -213 MHz against -311 MHz)
than IBM's fixed-frequency Eagle transmons. Its E_J/E_C of about 52 is in the
same regime as IBM's, but its charging energy is smaller. **What is not
comparable:** T1. The design's T1 is a loss-model bound set by the example's
deliberately pessimistic sapphire loss tangent (3e-5 to 8.6e-5). Device T1
reflects fabrication, TLS defects and packaging.

**Data provenance.** `qbraid devices get` and the SDK's `device.metadata()`
do not currently expose T1, T2 or qubit frequency, and live IBM properties
need the user's own IBM token. The table therefore uses the dated
(2025-02-26) calibration snapshots bundled in `qiskit-ibm-runtime`'s fake
backends. These are real calibrations but not live. Heron snapshots don't
publish frequency or anharmonicity (n/p).

## Reproduce

```bash
# 1. toolchain + Palace (about 24 min with -j3; about 5-8 min with -j16 on cpu-32v-128g)
export MAMBA_ROOT_PREFIX=/tmp/ose-mamba
micromamba create -y -p /tmp/ose-envs/palace -f environment.yml
PREFIX=/tmp/ose-envs/palace BUILD=/tmp/ose-build JOBS=3 ./build_palace.sh

# 2. everything else: run, validate, derive, simulate, compare, build viewer
NP=3 ./run_all.sh
```

Build gotchas, all handled in `build_palace.sh`:

- The conda-forge `make` 4.4 uses a fifo jobserver, and GSLIB's sub-build
  calls the system `gmake` 4.3, which fails with
  `invalid --jobserver-auth string 'fifo:...'`. The script forces
  `--jobserver-style=pipe`.
- The MPI wrappers are pointed at the conda compilers (`OMPI_CC`, `OMPI_CXX`,
  `OMPI_FC`) so every dependency uses one toolchain.
- Build on a large scratch disk. The env is about 4.9 GB and the build tree
  about 2.1 GB. On qBraid, use `/tmp` (overlay), not `$HOME`.

**Packaging as a qBraid environment (not done yet).** The installed
`palace-x86_64.bin` has an absolute RPATH (`/tmp/ose-envs/palace/lib`). It must
be rewritten to `$ORIGIN/../lib` (`patchelf --set-rpath`) before the env can
be relocated. The env has to ship its own Open MPI, because `palace -np N`
calls `mpirun` from `PATH`. The compilers can be dropped from the runtime env.

## Honest limits

- This is Palace's reference transmon (geometry by DeviceLayout.jl), not a
  new design. The pipeline is general, but only this geometry has been run.
- The mesh is the example's coarse mesh at order 2, with no adaptive mesh
  refinement. The Palace docs show AMR shifting results, so treat the
  frequencies as good to the coarse-mesh error, not to the digits shown.
- The Hamiltonian keeps two modes (qubit and resonator) and the feedline is
  modelled only through the port Q. Charge dispersion and the junction's
  higher harmonics beyond the cosine are ignored.
- Dynamics use a Lindblad model with T1 only. There is no flux or charge noise
  and no TLS.

## Next: needs an instance

| step | profile | estimate |
|---|---|---|
| AMR run (`transmon_amr.json`) or a finer mesh, 32 MPI ranks | `cpu-32v-128g` ($1.92/h) | 20-40 min, under $1.50 |
| Faster Palace build (`-j16`) | `cpu-32v-128g` | about 8 min, about $0.26 |
| GPU Palace (`PALACE_WITH_CUDA=ON`, cuDSS) | `gpu-a100-sxm` ($2.49/h) | build plus run about 1 h, about $2.50 |
| Design sweep (claw length → chi, cap gap → alpha), one run per instance | 4 × `cpu-32v-128g` | about $5 for 16 designs |

- **Front end:** generate layouts with Qiskit Metal or KQCircuits (IQM) and
  mesh them with gmsh, so a user can change the design, not just re-run the
  reference.
- **Closed loop on hardware:** spectroscopy and a T1 run on a real QPU,
  compared with the design. This spends credits and needs the user's IBM
  token from the Vault, so it requires the user's approval and a cost estimate
  first.
