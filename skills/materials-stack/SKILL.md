---
name: materials-stack
description: Run materials simulations on qBraid with universal machine-learned interatomic potentials (MACE-MP/MPA, ORB v3, EquiformerV3) plus a PySCF DFT anchor. Use for crystal stability screening (Matbench Discovery protocol), lattice constants and bulk moduli, phonons, MD of ionic conductors such as battery electrolytes, or an open alternative to VASP / Materials Studio screening. Covers model choice and weight licences, the exact benchmark protocol, the known failure modes (phonon softening, cutoff artefacts), GPU and disk sizing, and where quantum computing honestly fits.
metadata:
  version: "0.3.0"
  layer: "tool"
  status: "draft"
  verified: "2026-10-01"
---

# Materials stack: universal ML potentials on qBraid

Universal machine-learned interatomic potentials (MLIPs) do in seconds per structure what PBE DFT does in CPU-hours. They are trained on PBE(+U) data, so **PBE is their ceiling, not experiment.** Always say which one you are comparing against.

Start from the user's goal: if it is unclear whether they want the best solution or the best quantum solution, ask (see solution-router).

## Get the scripts

The worked example lives in `materials/` of https://github.com/qBraid/open-simulation-examples:

```bash
git clone --depth 1 https://github.com/qBraid/open-simulation-examples
```

All script paths below are relative to that repo. `materials/README.md` has every number and its wall time.

## Model choice (licence first)

| Model | `mace_mp(model=...)` / package | Matbench Discovery F1 (unique protos) | Checkpoint licence | Use it for |
|---|---|---|---|---|
| MACE-MP-0 medium | `"medium"` | 0.669 | MIT | Reproducing the literature; baseline |
| MACE-MP-0b3 medium | `"medium-0b3"` | (not on the leaderboard) | MIT | **Phonons and MD** (fixes most of MP-0's softening) |
| MACE-MPA-0 medium | `"medium-mpa-0"` | 0.852 | MIT | **Stability screening** among the MACE models |
| ORB v3 conservative-inf-mpa | `orb-models` (`pretrained.orb_v3_conservative_inf_mpa`) | 0.905 | Apache-2.0 | **Best open model that is easy to run**; screening |
| EquiformerV3+DeNS-OAM | research fairchem fork, torch 2.7 | 0.931 (#1) | MIT | Top accuracy; needs its own environment |

**Licences checked 2026-10-01 against the weights, not just the code.**
- MACE-MP-0, MP-0b3 and MPA-0: MIT (ACEsuit/mace-foundations table).
- ORB v3: Apache-2.0 (the orb-models LICENSE covers the models).
- EquiformerV3 checkpoints: MIT (Hugging Face `mirror-physics/equiformer_v3` card). Its training data, OMat24, sAlex and MPtrj, is CC-BY-4.0, which only requires attribution.
- Meta's own eSEN and eqV2 OMat checkpoints carry Meta's research licence. They are not used here.

**Do not use the ASL-licensed MACE models (OMAT-0, MATPES-*, MH-*) for commercial work.** The ASL is an academic licence. Check `ACEsuit/mace-foundations` before adopting a new checkpoint.

## Environment

No qBraid environment exists for this stack yet. Check `qbraid envs available` for one first. Otherwise build it as below, then package it with the manage-environments skill so the next instance installs it instead of rebuilding.

- `python3 -m venv env && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126 && pip install --no-cache-dir mace-torch ase pymatgen phonopy pyscf geometric`. Add `orb-models` for ORB.
- **Size.** About 6 GB, of which 3.6 GB is NVIDIA CUDA libraries. `orb-models` pulls in `triton` (0.9 GB) and `warp` (0.4 GB). Uninstall both if you don't use ORB with `compile=True`. Build on local disk (`/tmp`), not on a GPU instance's network home.
- **orb-models ≥ 0.7 changed its API.** `pretrained.orb_v3_conservative_inf_mpa(device=..., precision="float32-high", compile=False)` returns `(model, adapter)`. The calculator is `orb_models.forcefield.inference.calculator.ORBCalculator(model, adapter, device=...)`.
- **EquiformerV3** needs a separate environment (torch 2.7.1+cu128, the authors' fairchem fork, about 13 GB).
- **Model weights** download to `~/.cache/mace` and `~/.cache/orb` on first use (about 100 MB each).
- **Figshare blocks some instance IPs** and returns 0-byte files. Download Matbench data on the Lab pod and `scp` it to the instance.

## Benchmark protocol (Matbench Discovery, reproduced to about 1 meV/atom per structure)

1. Sample WBM `unique_prototype` structures with a stated seed. Read them from `wbm-initial-atoms.extxyz.zip` (figshare file 48169597). The summary is figshare 64706751.
2. Relax with FIRE and `FrechetCellFilter`, at most 500 steps, `fmax` 0.05 eV/Å for MACE and **0.02 for ORB v3**. Use float32 on GPU; the L4's FP64 rate is about 1/64 of FP32.
3. The MP2020 correction is additive per atom, so `e_form_pred = e_form_dft + (E_model − uncorrected_energy_from_cse)/n_sites`. Hull distance is `e_hull_pred = e_hull_dft + (e_form_pred − e_form_dft)`. Stable means ≤ 0.
4. Score F1, DAF, precision, recall and MAE, with a bootstrap CI on F1. **Also score the model authors' published predictions on the same IDs** (figshare `pred_file` in `models/<model>/*.yml`). That separates your pipeline error from sampling noise.

Scripts: `materials/scripts/discovery.py` and `score_discovery.py`.

| Model | n | F1 ours [95% CI] | F1, authors' predictions, same structures | s per relaxation (L4) |
|---|---|---|---|---|
| MACE-MPA-0 | 1000 | 0.876 [0.835, 0.913] | 0.872 | 0.84 |
| EquiformerV3+DeNS-OAM | 250 | 0.909 [0.837, 0.966] | 0.909 | 2.69 |

## Decision rules

- **Screening thousands of candidates:** ORB v3 or MACE-MPA-0. About 1–2 s per structure on an L4, so 1,000 structures in 20–35 minutes. Then confirm the top hits with DFT.
- **Lattice constants:** any of the MACE models reproduces PBE to about 0.45% (24-solid benchmark). It cannot beat PBE's 1.4% error against experiment.
- **Bulk moduli and phonons:** treat as qualitative. MACE-MP-0 underestimates Si optical phonons by about 27%; **MP-0b3 reduces this to about 5%**. Validate one known material first.
- **Heavy alkali metals (Cs, Rb) and long-bonded systems:** the 6 Å graph cutoff truncates neighbour shells, so E(V) has kinks and bulk moduli are meaningless. Scan E(V) widely before fitting.
- **MD of ionic conductors:** use float32 on GPU with a 2 fs step. About 400 atoms at tens of steps per second on an L4 means 20 ps in about 10–15 minutes. Run at least three temperatures for an Arrhenius fit. The ordered Materials Project cell is not the disordered experimental phase, so compare activation energies with care.
- **DFT anchor:** PySCF PBE/def2-TZVP (CPU) for molecules, or GPU4PySCF for larger systems. For periodic PBE use Quantum ESPRESSO or CP2K (GPL), not this stack.

## Compute on qBraid

Pick a profile from `qbraid compute list` and launch it through the qbraid-cloud-orchestration skill with `--auto-stop`; terminate it when done.

- **GPU:** `gpu-l4` is enough for every task here. Use `gpu-h100-sxm` only for full-leaderboard runs (257k structures, about 190 L4-hours for EquiformerV3) or 10k-atom MD. A `gpu-l4` instance's cgroup allows about 5 CPUs even though `nproc` shows 48.
- **CPU:** EOS for 24 solids takes about 30 s per model on one core, and PySCF small molecules take minutes. The subscription pod or a `cpu-8v-32g` instance is fine.
- **Disk:** keep environments on local `/tmp` and datasets on the network home. The WBM zip is 98 MB and is fine on either.

## Quantum: the short answer

**Runnable test today** (research-grade, no advantage claim). MLIPs and DFT are classical PBE-level methods; quantum computing enters only where PBE fails, at strongly correlated sites in catalysts, magnets and some battery cathodes. The credible hardware experiment: MLIP screen, then DFT, then pick the strongly correlated candidates, build a small active space, and run sample-based quantum diagonalization (SQD, `qiskit-addon-sqd`) on IBM Heron (156 qubits; needs the user's IBM token from the vault-credentials skill). Score it against classical CASCI or DMRG on the same active space, which is the anchor and must come first. Price the run with `qbraid devices get` and confirm with the user before submitting.

For the full answer and the experiment protocol, use the **quantum-readiness** skill (section "Chemistry and materials").

## Verification stamp

- 2026-10-01, `gpu-l4` instance (NVIDIA L4, driver 595, 5-CPU cgroup).
- torch 2.14.1+cu126, mace-torch 0.3.16, orb-models 0.7.0, ase 3.29.0, phonopy 4.7.2, pyscf 2.14.0; EquiformerV3 in its own env (torch 2.7.1+cu128).
- 112 GPU-minutes in 7 jobs plus about 10 single-core minutes for EOS and phonons: about $0.92 at $0.49/h.
- Full results and limits: `materials/README.md`.
