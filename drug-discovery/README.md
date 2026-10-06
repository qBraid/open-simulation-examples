# Drug discovery: co-folding and affinity ranking with open models

Predict how a small molecule binds its protein, rank a congeneric series by affinity, and check both against crystal structures, experiment, classical docking and the commercial FEP+ reference. Everything uses open weights (Boltz-2, MIT) on one NVIDIA L4. The interactive three.js viewer is `viewer.html`.

**Stands in for:** Schrödinger Glide and FEP+, MOE, and Discovery Studio, at the triage stage.

## What top-10% means here
| Task | Benchmark | Metric | Where the field sits |
|---|---|---|---|
| Pose prediction | PoseBusters V1 (428 complexes, after the 2021 cutoff), fixed 32-complex subset | % with ligand RMSD ≤ 2 Å (pocket-aligned), and % that are also PoseBusters-valid | AlphaFold3 76.4% and Chai-1 77% on the full V1 set (RMSD only); classical docking with the pocket given is about 50–60% |
| Affinity ranking | Schrödinger FEP+ JACS set: TYK2 and CDK2, 16 ligands each, same ligands as FEP+ | Spearman and Kendall vs experiment (bootstrap 95% CI), mean-centred RMSE | FEP+ (commercial, physics, hours of GPU per pair) is the accuracy reference |

The bar for "reached": pose success on the subset within sampling error of the published co-folding state of the art, and above classical docking on the same complexes. For affinity, we report where the open model sits relative to FEP+ on identical ligands, without claiming parity it doesn't have.

## Results
### Pose prediction: 32 PoseBusters V1 complexes
| Method | RMSD ≤ 2 Å | and PB-valid |
|---|---|---|
| **Boltz-2, this run** (co-folding, no pocket given) | 84% | 84% |
| GOLD docking (crystal protein and pocket given; PoseBusters paper) | 41% | 34% |
| VINA docking (crystal protein and pocket given; PoseBusters paper) | 50% | 44% |
| AlphaFold3 (Abramson et al., Nature 2024), full V1 set (published) | 76% | – |
| Chai-1 (Chai Discovery tech report, 2024), full V1 set (published) | 77% | – |

**Reached** against the bar. Boltz-2 gets 84% ± 6 (1σ, n=32) against published co-folding at 76%–77% on the full set, and it beats docking on the same complexes even though docking is handed the pocket.
- **Misses:** 5SAK_ZRY (6.5 Å, confidence 0.97), 7AMC_73B (7.8 Å, confidence 0.82), 7MS7_ZQ1 (19.2 Å, confidence 0.85), 7OSO_0V1 (2.8 Å, confidence 0.96), 8BRO_R7E (24.1 Å, confidence 0.91).
- **Confidence did not flag the large misses well.** They are high-confidence wrong pockets, which is the known failure mode of co-folding.

### Affinity ranking: Schrödinger FEP+ JACS series, identical ligands
| Target | n | Boltz-2 Spearman [95% CI] | Boltz-2 Kendall | FEP+ Spearman [95% CI] | FEP+ Kendall | RMSE (centred) Boltz-2 / FEP+ |
|---|---|---|---|---|---|---|
| TYK2 | 16 | 0.77 [0.40, 0.94] | 0.60 | 0.93 [0.73, 0.99] | 0.81 | 0.74 / 0.47 kcal/mol |
| CDK2 | 16 | 0.87 [0.64, 0.94] | 0.63 | 0.62 [0.17, 0.89] | 0.44 | 0.69 / 0.90 kcal/mol |

## Method
- **Subset rule (fixed before predicting):** from PoseBusters V1, keep complexes whose protein has ≤ 700 observed residues and whose ligand has ≤ 50 heavy atoms (320 of 428 qualify), then draw 32 with `random.Random(2026)`. The list is in `results/posebusters_subset.json`.
- **Inputs:** the protein as observed residues per chain, the ligand as SMILES from the crystal SDF. Cofactors and additional ligands are not modelled. MSAs come from the public ColabFold MMseqs2 server, precomputed on CPU (`prep_posebusters.py`, `prep_affinity.py`).
- **Boltz-2 settings:** 5 diffusion samples, 3 recycles, 200 sampling steps, physics steering potentials (`--use_potentials`), seed 0, and the top-ranked sample scored (`run_boltz.sh`).
- **Pose scoring (`eval_poses.py`):** superpose on pocket Cα atoms (residues within 10 Å of the crystal ligand), compute symmetry-aware ligand RMSD without refitting, then run PoseBusters `redock` checks against the superposed predicted protein. Success means RMSD ≤ 2 Å and PB-valid.
- **Docking baselines:** Vina and GOLD per-complex results released with the PoseBusters paper (Zenodo 8278563), restricted to the same 32 complexes. Docking gets the crystal protein and pocket, which is an easier task.
- **Affinity (`eval_affinity.py`):** Boltz-2 `affinity_pred_value` (log10 IC50/µM) converted with ΔG = 1.364·(v−6) kcal/mol. Ranking metrics don't depend on the conversion. FEP+ values come from Schrödinger's public benchmark (MIT), copied in `results/fep_plus_*.csv` and `fep_inputs/`.

## Reproduce
```bash
pip install -r requirements.txt            # Python 3.12, NVIDIA GPU with 24 GB or more
./run_all.sh                               # prep (CPU), Boltz-2 (GPU), scoring, viewer
```
See `skills/md-cofold-stack/SKILL.md` for the qBraid-specific setup: weights caching, CCD extraction, MSAs and GPU memory.

## Honest limits
- **32 complexes is a small sample:** ±8 percentage points (1σ). Published figures are on full sets.
- **Data leakage is possible.** PoseBusters V1 complexes postdate the 2021 PDB cutoff, but similar proteins and ligands can appear in training data. The PoseBusters paper and later work show success drops for novel pockets.
- **Cofactors and metals aren't modelled.** Where they shape the pocket, that hurts our numbers relative to AlphaFold3's protocol, which includes them.
- **The subset is filtered to ≤ 700 residues and ≤ 50 ligand heavy atoms** to fit an L4. Large multi-chain complexes are harder, so "84% vs 76–77%" is on an easier slice, not a head-to-head on the full set. A full-set run is about $8–10 (see Next).
- **The affinity series may be in the training data.** The TYK2 and CDK2 JACS series (published 2015) are public and widely used, and binding data for them is in ChEMBL, which Boltz-2's affinity head was trained on. Read the Boltz-2 vs FEP+ comparison as "competitive on well-known series", not as evidence about novel chemistry. Affinity runs used 1 structure sample per ligand to save GPU time.
- **The affinity head ranks;** it doesn't replace FEP for lead optimisation. The IC50 ≈ Kd conversion is approximate.
- **No MD stability check.** It needs OpenFF or GAFF ligand parameters (a conda stack), which didn't fit the shared instance's disk and budget. It's listed as next.

## Next, needs more compute
- **The full PoseBusters V1/V2 sets:** about 2–3 min per complex on an L4, about $8–10 for all 428.
- **Relative FEP with OpenFE** on TYK2 as an open physics comparison: about 2–4 GPU-hours per edge set on an L4, roughly $10–20.
- **OpenMM MD** (50 ns) of the top poses to separate stable from transient binding modes.

## Where this fits
This is the Layer-1 `md-cofold-stack` pair in the qBraid skills design record. The pharma sector router (D1) points here for pose and affinity triage, and to the chemistry stack (PySCF, active-space VQE) for electronic structure.

## Verification stamp
- date: 2026-10-01
- machine: qBraid gpu-l4 instance, shared (NVIDIA L4 24 GB, driver 595, ~5 vCPU cgroup)
- env: Python 3.12, boltz 2.2.1, torch 2.14.1+cu130, posebusters 0.6.5, rdkit, gemmi
- gpu_minutes: 171.4
- gpu_jobs: 5
- cost_usd_gpu_share: 1.4
- note: CPU prep (MSAs via the public ColabFold server) ran at low priority on the same box; viewer built on the subscription pod at no extra cost
