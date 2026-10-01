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
`qbraid-canvas viewer.html`. It is a single 2.3 MB file, and three.js loads
from jsDelivr.


## v2: a design aimed at state-of-the-art T1 (2026-10-01)

**Bar (top-10% for this field).** A transmon whose Hamiltonian is a modern
fixed-frequency device (f01 4.5–5 GHz, alpha about -300 MHz, E_J/E_C 40–55,
chi/kappa about 0.5) and whose **predicted geometry-limited T1** sits within the
range of IBM's production medians (Eagle/Heron, 145–287 µs). Beating that
range would need to approach the tantalum records (median 0.45 ms, best
1.68 ms; Bland et al., Nature 647, 343 (2025)).

**Verdict.**
- **Hamiltonian targets: reached.** f01 4.800 GHz, alpha -290 MHz,
  E_J/E_C 50.4, g 93 MHz, chi 0.64 MHz,
  chi/kappa 0.50.
- **Predicted T1 in line with IBM medians: reached.** 146–245 µs
  for Ta on annealed sapphire, against IBM medians of 145–287 µs.
- **Tantalum-record class (above 0.5 ms): not reached.** The surface-loss limit
  alone is 178–352 µs, and planar
  edge participation is the floor. Getting past it needs substrate trenching
  or a 3D or flip-chip geometry, which is out of scope here.

**This T1 is a prediction under stated assumptions, not a measurement.**

### What changed from v1

| | v1 (Palace example) | v2 final |
|---|---|---|
| Island | 24 × 620 µm, 30 µm trench | **240 × 190 µm, 150 µm trench** |
| Surface participation p_MS+p_SA+p_MA | 8.49e-04 | **4.71e-04** |
| Predicted T1, Ta/sapphire (same targets) | 94–168 µs | **146–245 µs** |
| E_C / alpha | 236 MHz / -268 MHz | 252 MHz / -290 MHz |
| Junction | | L_J 12.85 nH, R_n target 11.0 kOhm (Ambegaokar–Baratoff, Al) |
| Readout | | 7.0 GHz lambda/4, kappa = 2chi = 1.29 MHz, Q_ext 5429, C_kappa 7.7 fF |
| Purcell | | 69 µs unfiltered, then **1.9 ms** with a Q=30 bandpass Purcell filter |

### Method

The two-step surface-participation method of Wang et al., APL 107, 162601 (2015),
with open tools (`loss/`):

1. **`edge2d.py`**: a 2D finite-element model (scikit-fem, P2) of a thin-film
   edge, giving the energy in 3-nm MS, MA and SA layers per K² as a function of
   the local gap. It converges to 0.03% under 4× mesh refinement.
2. **`geom3d.py` + Palace `Electrostatic`**: a 3D half model of the island,
   trench and readout claw, with metal as zero-thickness sheets and one terminal
   per conductor. Fields are normalised by Palace's terminal excitation voltage
   (`terminal-V.csv`).
3. **`post3d.py`**: the coarse 3D field on pad interiors and bare substrate,
   with edge strength K from the surface charge 1–4 µm from each edge, times
   the 2D integrals. Per-edge densities drive the viewer's "loss beads".
4. **`design_v2.py`**: lumped-oscillator Hamiltonian (E_C from C_sigma, E_J
   solved for the target f01), exact transmon–resonator diagonalisation for chi,
   the readout design and the T1 budget.

**Sweep** (`results/v2/sweep.json`): 11 island widths and trench gaps, each
rescaled to the same C_sigma. Participation falls monotonically with a wider
island and a wider trench: 9.2e-4 at W24/G30, 3.5e-4 at W240/G150. Two final
candidates were then solved at the target capacitance. The one chosen best
matches alpha ≈ -300 MHz at chi/kappa = 0.5.

### Assumptions behind the T1 number

- **Interface layers:** t = 3 nm, eps = 10 for MS, MA and SA (the Wang and Ganjam convention).
- **Ta on annealed sapphire:** surface loss tangent 3.4e-4 on p_MS+p_SA+p_MA,
  bulk 2.6e-8 (Ganjam et al., Nat. Commun. 15, 3687 (2024)). The Al/lift-off
  alternative uses Wang 2015 (tan_MS-weighted 2.6e-3) and gives
  48–92 µs for the same geometry.
- **Raw vs calibrated:** for the same pads our pipeline gives about
  2.0× Wang's participation, so we quote both raw (conservative)
  and calibrated (k = 0.51).
- **Junction leads:** Wang's measured lead participation is added. Junction
  TLS, quasiparticles, radiation, package modes and flux noise are **not modelled**.
- **Purcell:** assumes a Q=30 bandpass filter. Without one the readout limits T1 to 69 µs.

### Validation against a published device (`results/v2/validation_wang2015.json`)

| check | ours | reference | verdict |
|---|---|---|---|
| 2D edge model: 4x finer corner mesh | 0.03 % change | < 1 % | match |
| 2D edge strength K vs analytic thin slot (g=500 um) | 0.993 | 1 (h->0) | match |
| 3D field follows K/sqrt(u), 1.4-10 um from edges | flat to +-2 % | theory: flat | match |
| 3D mesh refinement: edge elements 1.0 -> 0.5 um (Design A) | +2.0 % | < 5 % | match |
| Edge band x0 = 4 vs 8 um (same 3D mesh) | +19 % | 0 % ideal | close |
| Wang 2015 Design A, pad p_MS (Table S1) | 1.64e-4 | 0.83e-4 | 2.0x high |
| Design A T1 from our p (Wang loss model) | 48 us | 66-95 us measured | conservative |
| v1 Palace eigenmode vs Palace reference (8 qty) | 8/8, f to 9e-10 | regression data | match |
| v1 E_C: electrostatic LOM vs eigenmode EPR | 226 MHz | 213 MHz | 6 % high |

The 2D edge model, the 3D field's 1/sqrt(u) edge law and the 3D mesh
convergence check out. The open item is the **2× offset from Wang's published
participation** for their Design A. Their pad gap and film thickness are not
fully specified, and the edge-band choice (x0 = 4 vs 8 µm) moves our number by
19%. This offset is why every T1 is quoted as a raw-to-calibrated range. As a
consequence, our raw prediction for Design A (48 µs) is conservative against
its measured 66–95 µs.

### Fab-ready output

`results/v2/mask/final_qubit.gds` (and `v1_qubit.gds`) is the exact analysed
geometry as a GDS mask: metal on layer 1, a junction placeholder on layer 2,
parameters on a text label. The Josephson junction itself is a separate
e-beam / Dolan-bridge step. Target R_n = 11.0 kOhm at room
temperature, to be checked by probe-station resistance before cooldown.

### Viewer

`viewer.html` (2.3 MB, three.js from jsDelivr):
- **Chip & modes:** the v1 chip from the full 3D eigenmode run, in PBR metal on
  sapphire, with a glowing |E| for each mode and a morph between qubit and
  resonator modes.
- **Loss map & evolution:** extruded metal for every design point, a
  slider/autoplay morph through the sweep to the final design, "loss beads"
  sized and coloured by participation per µm of edge, and the final design's
  qubit-mode field as a glow.
- **Panels:** the energy-level diagram with dispersive pull, the sweep chart, T1
  against IBM devices, the T1 budget and the validation table.
- **Guided tour** (6 steps) and light/dark themes. Screenshots are in `results/viewer_*.png`.

### Reproduce (on a CPU instance, not a shared pod)

```bash
python loss/pipeline.py table 0.2 10.34 4.0 runs/table.json
python loss/pipeline.py point runs/final '{"W":240,"G":150,"L":190}' runs/table.json --fieldmap
python loss/pipeline.py point runs/v1    '{"W":24,"G":30,"L":620}'   runs/table.json
# sweep: W in 24..320, G in 30..150 at L=400 -> runs/sw_W<W>_G<G>
QD_RUNS=runs python loss/make_results.py <final_run_dir> && python loss/make_validation.py
python build_viewer_v2.py
```

> **Verified 2026-10-01.** Palace v0.18.1 electrostatics: 14 design points plus
> 2 final candidates, each about 2–3 min on 2 MPI ranks on the shared qBraid
> gpu-l4 pool box (cgroup about 5 CPUs). Total about 37 CPU-minutes, about $0.30
> of pool time.

**Not done in v2.** A 3D eigenmode run of the final design: the pod run was
lost to a pod restart, and the pool's 60-min job cap with 2 threads is too
tight. The LOM E_C was 6% higher than the eigenmode/EPR value on v1, so expect
a few-percent frequency correction. That run costs about 1 h on `cpu-32v-128g`
($1.92).

---

## v1 results (Palace reference transmon)


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
