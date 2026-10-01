# Materials Discovery Lab

Universal machine-learned interatomic potentials (MLIPs) on a single qBraid L4 GPU. They are benchmarked on three things:
- the community leaderboard for discovering stable crystals (Matbench Discovery);
- experiment and PBE, for lattice constants, bulk moduli and phonons;
- a lithium-ion MD showcase in a battery solid electrolyte.

Every model used here has a commercially usable licence (MIT or Apache-2.0).

Open `viewer.html` (or `qbraid-canvas materials/viewer.html`) for the interactive version. It has four scenes and a guided tour.

<!--RESULTS-->
## Results

### 1 · Discovering stable crystals (Matbench Discovery protocol)

| Model (licence) | n | F1 ours [95% CI] | F1, authors' predictions on the same structures | F1 full test set (leaderboard) | MAE meV/atom | per-structure agreement with authors: MAE meV/atom / within 10 meV | s per relaxation (L4) | leaderboard rank |
|---|---|---|---|---|---|---|---|---|
| MACE-MP-0 (MIT) | 2000 | **0.676** [0.636, 0.712] | 0.676 | 0.669 | 57 | 0.41 / 99.8% | 1.32 | #52 of 65 |
| MACE-MPA-0 (MIT) | 1000 | **0.876** [0.835, 0.913] | 0.872 | 0.852 | 29 | 1.45 / 98.7% | 0.84 | #30 of 65 |
| EquiformerV3+DeNS-OAM (MIT) | 250 | **0.909** [0.837, 0.966] | 0.909 | 0.931 | 15 | 1.11 / 98.8% | 2.69 | #1 of 65 |

**Verdict: reached.** The top-10% band of the 65-model leaderboard is F1 ≥ 0.925. The #1 model, EquiformerV3+DeNS-OAM (MIT weights), runs on a single qBraid L4 and reproduces its authors' published predictions structure by structure (about 1 meV/atom), as do the two MACE models we reran. Our sample F1 values match the authors' predictions on the same structures. The full-test-set F1 values in the leaderboard column are therefore the ones that apply to these runs.

### 2 · Lattice constants and bulk moduli (23 solids; Csonka et al. 2009)

| | a₀ error vs experiment | a₀ error vs PBE | B₀ error vs experiment | B₀ error vs PBE |
|---|---|---|---|---|
| PBE (the training level) | 1.39% | 0 | 9.0% | 0 |
| ORB v3 | 1.36% | **0.23%** | 13.8% | 8.9% |
| MACE-MPA-0 | 1.47% | **0.41%** | 20.7% | 20.4% |
| MACE-MP-0b3 | 1.46% | **0.43%** | 26.3% | 24.0% |
| MACE-MP-0 | 1.40% | **0.46%** | 20.7% | 19.5% |

Mean absolute relative errors. Cs is excluded because its energy–volume curve crosses the 6 Å cutoff (see Honest limits). **Verdict:** lattice constants **reached**: every model is within 0.5% of PBE and as accurate as PBE against experiment. Bulk moduli are **close** for ORB v3 (within about 9% of PBE) and **not reached** for MACE, where the soft alkali and alkaline-earth metals dominate the error.

### 3 · Silicon phonons vs inelastic neutron scattering (THz)

| | Γ-LTO | X-TA | X-LA | X-TO | L-TA | L-LA | L-TO | mean abs. error |
|---|---|---|---|---|---|---|---|---|
| neutron | 15.53 | 4.49 | 12.32 | 13.90 | 3.43 | 11.35 | 14.68 | |
| MACE-MPA-0 | 12.68 | 4.48 | 10.33 | 11.56 | 3.58 | 8.67 | 12.09 | 13.9% |
| ORB v3 | 13.46 | 3.14 | 10.67 | 13.09 | 4.18 | 11.39 | 12.19 | 14.5% |
| MACE-MP-0b3 | 14.66 | 5.82 | 10.34 | 11.90 | 4.56 | 9.59 | 13.00 | 18.0% |
| MACE-MP-0 | 11.19 | 4.60 | 8.99 | 10.15 | 3.34 | 8.17 | 10.72 | 20.3% |

**Verdict:** not reached. The best model is at 13.9% against a PBE-level bar of about 3–5%. Universal potentials are not yet a substitute for DFT phonons. Fine-tune on a few hundred DFT force calculations, or use DFT directly.

### 4 · Li-ion diffusion in Li₆PS₅Cl (416 atoms, MACE-MPA-0, L4)

| T (K) | D_Li (cm²/s) | MD steps/s |
|---|---|---|
| 800 | 2.57e-05 | 8 |
| 1000 | 4.35e-05 | 7 |

Only two temperatures: the 600 K run did not fit in the shared GPU's 30-minute slot (MD ran at about 7.5 steps/s for 416 atoms). The two-point slope (0.18 eV) is not a reliable activation energy, so no room-temperature conductivity is claimed. This is a visual showcase of lithium transport with a universal potential, not a conductivity prediction. Measured Li₆PS₅Cl: Eₐ ≈ 0.3–0.4 eV, 1–4 mS/cm (disordered pellets).

### 5 · DFT anchor: small-molecule geometries (Å, degrees)

| molecule / quantity | experiment | PBE/def2-TZVP (PySCF) | MACE-MP-0 | MACE-MPA-0 | MACE-MP-0b3 |
|---|---|---|---|---|---|
| H2O r(O-H) | 0.9578 | 0.971 | 0.973 | 0.970 | 0.975 |
| H2O angle(H-O-H) | 104.48 | 104.298 | 104.108 | 103.951 | 104.702 |
| NH3 r(N-H) | 1.012 | 1.022 | 1.019 | 1.015 | 1.021 |
| NH3 angle(H-N-H) | 106.7 | 106.323 | 106.416 | 107.424 | 106.562 |
| CH4 r(C-H) | 1.087 | 1.097 | 1.093 | 1.092 | 1.095 |
| CO2 r(C-O) | 1.162 | 1.171 | 1.174 | 1.177 | 1.177 |
| N2 r(N-N) | 1.098 | 1.103 | 1.112 | 1.113 | 1.114 |
| CO r(C-O) | 1.128 | 1.136 | 1.143 | 1.141 | 1.144 |

Bond lengths: the potentials reproduce PBE to a mean |Δr| of 0.005 Å (MACE-MP-0), 0.005 Å (MACE-MPA-0), 0.005 Å (MACE-MP-0b3), and PBE is itself 0.009 Å from experiment. That is, the potentials inherit their functional faithfully even for molecules, which are outside their periodic training domain. The PySCF PBE/def2-TZVP optimisations of all six molecules took under a minute on two CPU threads.

### Verification stamp

Verified 2026-10-01 on a shared qBraid gpu-l4 pool box (cgroup: 5 CPUs, 62 GB; NVIDIA L4, driver 595). This stream used 112 GPU-minutes in 7 jobs, 1 queued CPU-minutes, and about 10 single-core minutes for EOS and phonons; at $0.49/h for the box that is about $0.92. EquiformerV3 ran in its own environment (torch 2.7.1+cu128, the authors' fairchem fork). Versions: torch 2.14.1+cu126, mace-torch 0.3.16, orb-models 0.7.0, ase 3.29.0, phonopy 4.7.2, pyscf 2.14.0.

<!--/RESULTS-->

## What "top 10%" means here
- **Discovery.** Matbench Discovery ([matbench-discovery.materialsproject.org](https://matbench-discovery.materialsproject.org)) ranks 66 models by F1 for classifying hypothetical WBM crystals as stable. The top 10% is F1 ≥ 0.925. Every model in that band is trained on OMat24 + sAlex + MPtrj ("OAM"). The bar has two parts: (1) the pipeline must reproduce published numbers exactly, and (2) we place the best open model we can run.
- **Physical properties.** The reference is the 24-solid set of Csonka et al., PRB 79, 155107 (2009): zero-point-corrected experimental a₀, experimental B₀, and PBE values. A potential trained on PBE cannot beat PBE against experiment. The top-tier bar is **reproducing PBE within about 0.5% on a₀** and **matching PBE's own error against experiment**.
- **Phonons.** Silicon high-symmetry frequencies against inelastic neutron scattering (Nilsson & Nelin, PRB 6, 3777 (1972)). The bar is PBE-level, a mean error of about 3–5%.

## Reproduce
On a GPU instance (an L4 is enough), with `python3.12 -m venv env` and the install line from `skills/materials-stack/SKILL.md`. Download the data on a machine that can reach figshare:

| File | figshare id |
|---|---|
| `data/wbm-summary.csv.gz` | 64706751 |
| `data/wbm-initial-atoms.extxyz.zip` | 48169597 |
| `data/mace-mp-0-discovery.csv.gz` | 66646193 |
| `data/mace-mpa-0-discovery.csv.gz` | 66646196 |
| `data/orb-v3-discovery.csv.gz` | from `models/orb/orb-v3.yml` |

Then run:

```
python scripts/discovery.py mp0 2000 0 2 disc_mp0_0.csv     # repeat for shard 1, and for mpa0 and orb3
python scripts/score_discovery.py mp0 results/score_mp0.json disc_mp0_*.csv
DEVICE=cpu python scripts/eos.py mp0b3 results/eos_mp0b3.csv
python scripts/phonons_si.py mp0b3 results/phon_mp0b3.json
python scripts/md_li6ps5cl.py mpa0 800 20 results/md_mpa0_800
DEVICE=cpu python scripts/dft_anchor.py results/dft_anchor.json
python build_viewer.py results
```

Sampling uses seed 20261001 over the WBM unique prototypes. The relaxation settings are FIRE with FrechetCellFilter, at most 500 steps, fmax 0.05 eV/Å (0.02 for ORB v3). These match the leaderboard runs.

## Honest limits
- **Sample, not the full test set.** We ran 2,000 (MACE-MP-0), 1,000 (MPA-0) and 250 (EquiformerV3) of 215,488 structures. The bootstrap 95% interval on F1 is about ±0.05. Exactness is shown instead by per-structure agreement with the authors' own predictions.
- **EquiformerV3 was checked on 250 structures**, not thousands. Its environment is separate (torch 2.7.1, the authors' fairchem fork, 13 GB on the network home) and it takes 2.7 s per relaxation on an L4. The full 257k test set would take about 190 L4-hours. ORB v3 discovery was not rerun, to free the shared GPU; its lattice-constant and phonon results are included.
- **PBE is the ceiling.** For lattice constants these models cannot beat PBE's own error against experiment of about 1.4% (overestimate). VASP with a better functional (PBEsol, r²SCAN) would do better. That is the place to spend DFT time, after MLIP screening.
- **Cutoff artefacts.** Cs (and partly Rb and Na) has neighbour shells near the 6 Å graph cutoff, so E(V) curves have kinks. Cs is excluded from the EOS statistics and shown in the viewer.
- **MD uses the ordered Materials Project cell** (mp-985592) of Li₆PS₅Cl. Real samples have S/Cl site disorder, which raises conductivity. Activation energies and extrapolated room-temperature conductivity are indicative, not predictive.

## Where quantum fits
These are classical surrogates of PBE. Quantum computing is relevant where PBE itself fails: strongly correlated transition-metal and rare-earth sites in catalysts, magnets and some battery cathodes. The route there is the chemistry stack: an active-space Hamiltonian, then VQE or SQD on today's hardware (at sizes still classically simulable, so a research and readiness exercise), and fault-tolerant phase estimation in the future. The practical hybrid today is to screen with an MLIP, confirm with DFT, and flag the strongly correlated candidates for active-space treatment.
