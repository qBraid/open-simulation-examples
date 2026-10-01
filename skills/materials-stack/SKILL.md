---
name: materials-stack
description: Run materials simulations on qBraid with universal machine-learned interatomic potentials (MACE-MP/MPA, ORB v3) plus a PySCF DFT anchor. Use for crystal stability screening (Matbench Discovery protocol), lattice constants and bulk moduli, phonons, and MD of ionic conductors such as battery electrolytes. Covers model choice and licences, the exact benchmark protocol, the known failure modes (phonon softening, cutoff artefacts), and GPU and disk sizing on qBraid.
---

# Materials stack: universal ML potentials on qBraid

Universal machine-learned interatomic potentials (MLIPs) do in seconds per structure what PBE DFT does in CPU-hours. They are trained on PBE(+U) data, so **PBE is their ceiling, not experiment.** Always say which one you are comparing against.

## Model choice (licence first)

| Model | `mace_mp(model=...)` / package | Matbench Discovery F1 (unique protos) | Checkpoint licence | Use it for |
|---|---|---|---|---|
| MACE-MP-0 medium | `"medium"` | 0.669 | MIT | Reproducing the literature; baseline |
| MACE-MP-0b3 medium | `"medium-0b3"` | (not on the leaderboard) | MIT | **Phonons and MD** (fixes most of MP-0's softening) |
| MACE-MPA-0 medium | `"medium-mpa-0"` | 0.852 | MIT | **Stability screening** among the MACE models |
| ORB v3 conservative-inf-mpa | `orb-models` (`pretrained.orb_v3_conservative_inf_mpa`) | 0.905 | Apache-2.0 | **Best open model that is easy to run**; screening |
| EquiformerV3+DeNS-OAM | research fairchem fork, torch 2.4 | 0.931 (#1) | MIT | Top accuracy; needs its own environment (see below) |

**Do not use the ASL-licensed MACE models (OMAT-0, MATPES-*, MH-*) for commercial work.** The ASL is an academic licence. Check `ACEsuit/mace-foundations` before adopting a new checkpoint.

## Install (verified)
- `python3 -m venv env && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126 && pip install --no-cache-dir mace-torch ase pymatgen phonopy pyscf geometric`. Add `orb-models` for ORB.
- **Size.** About 6 GB, of which 3.6 GB is NVIDIA CUDA libraries. `orb-models` pulls in `triton` (0.9 GB) and `warp` (0.4 GB). Uninstall both if you don't use ORB with `compile=True`.
- **orb-models ≥ 0.7 changed its API.** `pretrained.orb_v3_conservative_inf_mpa(device=..., precision="float32-high", compile=False)` returns `(model, adapter)`. The calculator is `orb_models.forcefield.inference.calculator.ORBCalculator(model, adapter, device=...)`.
- **Model weights** download to `~/.cache/mace` and `~/.cache/orb` on first use (about 100 MB each).
- **Figshare blocks some instance IPs** and returns 0-byte files. Download Matbench data on the Lab pod and `scp` it across.

## Benchmark protocol (Matbench Discovery, reproduced to 0.3 meV/atom)
1. Sample WBM `unique_prototype` structures with a stated seed. Read them from `wbm-initial-atoms.extxyz.zip` (figshare file 48169597). The summary is figshare 64706751.
2. Relax with FIRE and `FrechetCellFilter`, at most 500 steps, `fmax` 0.05 eV/Å for MACE and **0.02 for ORB v3**. Use float32 on GPU; the L4's FP64 rate is about 1/64 of FP32.
3. The MP2020 correction is additive per atom, so `e_form_pred = e_form_dft + (E_model − uncorrected_energy_from_cse)/n_sites`. Hull distance is `e_hull_pred = e_hull_dft + (e_form_pred − e_form_dft)`. Stable means ≤ 0.
4. Score F1, DAF, precision, recall and MAE, with a bootstrap CI on F1. **Also score the model authors' published predictions on the same IDs** (figshare `pred_file` in `models/<model>/*.yml`). That separates your pipeline error from sampling noise.

Scripts: `materials/scripts/discovery.py` and `score_discovery.py`.

## Decision rules
- **Screening thousands of candidates:** ORB v3 or MACE-MPA-0. About 1–2 s per structure on an L4, so 1,000 structures in 20–35 minutes. Then confirm the top hits with DFT.
- **Lattice constants:** any of the MACE models reproduces PBE to about 0.45% (24-solid benchmark). It cannot beat PBE's 1.4% error against experiment.
- **Bulk moduli and phonons:** treat as qualitative. MACE-MP-0 underestimates Si optical phonons by about 27%; **MP-0b3 reduces this to about 5%**. Validate one known material first.
- **Heavy alkali metals (Cs, Rb) and long-bonded systems:** the 6 Å graph cutoff truncates neighbour shells, so E(V) has kinks and bulk moduli are meaningless. Scan E(V) widely before fitting.
- **MD of ionic conductors:** use float32 on GPU with a 2 fs step. About 400 atoms at tens of steps per second on an L4 means 20 ps in about 10–15 minutes. Run at least three temperatures for an Arrhenius fit. The ordered Materials Project cell is not the disordered experimental phase, so compare activation energies with care.
- **DFT anchor:** PySCF PBE/def2-TZVP (CPU) for molecules, or GPU4PySCF for larger systems. For periodic PBE use Quantum ESPRESSO or CP2K (GPL), not this stack.

## Compute on qBraid
- **GPU:** `gpu-l4` is enough for every task here. Use an H100 only for full-leaderboard runs (257k structures) or 10k-atom MD.
- **CPU:** EOS for 24 solids takes about 30 s per model on one core, and PySCF small molecules take minutes. Use the subscription pod or a small CPU instance.
- **Disk:** keep envs on local `/tmp` and datasets on the network home. The WBM zip is 98 MB and is fine on either.

## Quantum angle (honest)
These are classical PBE-level surrogates. Quantum computing enters where PBE fails: strongly correlated sites in catalysts and magnets. There the route is the chemistry-stack, with an active space, then VQE/SQD today (classically simulable sizes, as research) and fault-tolerant phase estimation later. A practical hybrid today is to screen with an MLIP, confirm with DFT, and flag strongly correlated candidates for active-space methods.

## Verification stamp
Verified on 2026-10-01 on a qBraid `gpu-l4` (5-CPU cgroup) with torch 2.14.1+cu126, mace-torch 0.3.16 and orb-models 0.7.0. See `materials/README.md` for every number and its wall time.
