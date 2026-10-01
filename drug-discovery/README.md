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
RESULTS_PLACEHOLDER

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
- **The affinity head ranks;** it doesn't replace FEP for lead optimisation. The IC50 ≈ Kd conversion is approximate.
- **No MD stability check.** It needs OpenFF or GAFF ligand parameters (a conda stack), which didn't fit the shared pool's disk and budget. It's listed as next.

## Next, needs more compute
- **The full PoseBusters V1/V2 sets:** about 1 min per complex on an L4, about $3–4.
- **Relative FEP with OpenFE** on TYK2 as an open physics comparison: about 2–4 GPU-hours per edge set on an L4, roughly $10–20.
- **OpenMM MD** (50 ns) of the top poses to separate stable from transient binding modes.

## Where this fits
This is the Layer-1 `md-cofold-stack` pair in the qBraid skills design record. The pharma sector router (D1) points here for pose and affinity triage, and to the chemistry stack (PySCF, active-space VQE) for electronic structure.

## Verification stamp
STAMP_PLACEHOLDER
