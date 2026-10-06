---
name: md-cofold-stack
description: Protein-ligand pose prediction and affinity ranking on qBraid GPUs with open models (Boltz-2, Chai-1) plus classical anchors (Vina/GOLD docking results, FEP+ reference values, OpenMM). Use for co-folding a protein with a small molecule, ranking a congeneric series, benchmarking against PoseBusters or the FEP+ sets, rendering poses in three.js, or an open alternative to Glide / FEP+ triage. Covers install, weights caching, MSA handling, GPU memory limits, the pocket-aligned RMSD + PoseBusters scoring protocol, honest comparison rules, and where quantum computing fits.
metadata:
  version: "0.3.0"
  status: "provisional"
  verified: "2026-10-01"
---

# Co-folding and affinity stack (Boltz-2 first)

Open co-folding models for pose triage and affinity ranking, always scored against a classical anchor.

Start from the user's goal: if it is unclear whether they want the best solution or the best quantum solution, ask (see solution-router).

## Get the scripts

The worked example lives in `drug-discovery/` of https://github.com/qBraid/open-simulation-examples:

```bash
git clone --depth 1 https://github.com/qBraid/open-simulation-examples
```

All script paths below are relative to that repo: `prep_posebusters.py`, `prep_affinity.py`, `run_boltz.sh`, `eval_poses.py`, `eval_affinity.py`, and `run_all.sh` for the whole pipeline.

## Decision rules

| Question | Default | When to switch |
|---|---|---|
| Where does this ligand bind, in what pose? | **Boltz-2** co-folding (MIT, weights public) | Chai-1 (Apache) as a second opinion; classical docking (Vina, GNINA) when the pocket is known and you need hundreds of thousands of ligands |
| Rank a congeneric series | **Boltz-2 affinity head** for triage (a few minutes per ligand on an L4) | Relative FEP (OpenFE, open; FEP+, commercial) for lead optimisation. It is hours per ligand pair and still the accuracy reference |
| Is the pose physically sane? | **PoseBusters** checks against the predicted protein | Short OpenMM MD to check stability |
| What will the user see? | three.js ribbons, ball-and-stick, pocket surface, pose morph (`drug-discovery/viewer.html`); render with the agent-canvas skill | Mol* for interactive inspection |

Always report against a classical anchor: docking success on the same complexes (the PoseBusters paper releases per-complex Vina and GOLD results, Zenodo 8278563), and FEP+ predictions on the same ligands (Schrödinger's public benchmark, MIT). Without that, a co-folding number means nothing.

## Environment

No qBraid environment exists for this stack yet. Check `qbraid envs available` for one first. Otherwise build it from `drug-discovery/requirements.txt` (Python 3.12, NVIDIA GPU with 24 GB or more), then package it with the manage-environments skill for reuse.

- **Install:** `pip install -r drug-discovery/requirements.txt` (boltz 2.2.1, posebusters, rdkit, gemmi) in a venv on local disk (`/tmp`). It pulls torch with CUDA 13; driver 595 on a `gpu-l4` instance is fine. The venv is about 6 GB, nearly all torch and CUDA wheels.
- **No cuEquivariance kernels are installed,** so pass `--no_kernels` or Boltz fails at the triangle-attention import.
- **Weights:** about 4.3 GB (`boltz2_conf.ckpt` 2.3 GB plus `boltz2_aff.ckpt` 2.1 GB). Download them once to a large disk and symlink them into the `--cache` dir.
- **Don't let Boltz extract the full CCD `mols.tar` onto a network filesystem.** It is 45k small files and crawls. Extract only the canonical residues (`boltz.data.const.tokens` plus UNK) into a local cache dir and symlink `mols.tar` alongside. Ligands given as SMILES don't need CCD entries.

## Running it

- **Generate MSAs on CPU first** with `boltz.data.msa.mmseqs2.run_mmseqs2([seq], prefix)` against the public ColabFold server (about 2 s per chain), and write `.a3m` files referenced from the YAML. Otherwise `--use_msa_server` holds the GPU while it waits on the network.
- **L4 (24 GB) memory:** keep complexes at 700 observed residues or fewer, with 5 diffusion samples. Multi-chain targets (for example CDK2/cyclin A in the FEP+ set) need one `protein:` entry and MSA per chain.
- **Batch GPU work:** one YAML directory per batch, with outputs per batch, so a lost instance costs one batch, not the run. Copy results back after each batch.
- **Use `--use_potentials`** (physics steering): it targets exactly the PoseBusters validity failures.
- **Compute:** a `gpu-l4` instance from `qbraid compute list`, launched through the qbraid-cloud-orchestration skill with `--auto-stop`, terminated when done. About 2–3 min per complex: the full PoseBusters V1 set (428) is about $8–10.

## Scoring protocol (do not improvise)

1. Superpose the predicted protein onto the crystal using CA atoms of residues within 10 Å of the crystal ligand.
2. Compute symmetry-aware ligand heavy-atom RMSD with RDKit `CalcRMS`, with no refitting. Assign bond orders from the input SMILES first (`AssignBondOrdersFromTemplate`).
3. Run PoseBusters `redock` checks with the predicted ligand, the crystal ligand, and the superposed predicted protein.
4. **success = RMSD ≤ 2 Å and PB-valid.** Report RMSD-only too, because published co-folding numbers are often RMSD-only.
5. Fix the subset rule and seed before predicting. Report the binomial error (±8 points at n=32).

For affinity: Spearman and Kendall with bootstrap CIs, plus a mean-centred RMSE. Boltz-2's `affinity_pred_value` is log10(IC50/µM); dG ≈ 1.364·(v−6) kcal/mol.

| Verified 2026-10-01 | Result | Classical anchor |
|---|---|---|
| Poses, 32 PoseBusters V1 complexes (≤ 700 residues, ≤ 50 ligand heavy atoms, seed 2026) | 84% RMSD ≤ 2 Å and PB-valid (±6, 1σ) | Vina 44%, GOLD 34% PB-valid, with the pocket given; AlphaFold3 76%, Chai-1 77% on the full set (RMSD only) |
| Affinity, TYK2 / CDK2 (16 ligands each) | Spearman 0.77 / 0.87 | FEP+ 0.93 / 0.62 on the same ligands |

Large misses came with high confidence (wrong pocket), the known co-folding failure mode. Details and limits: `drug-discovery/README.md`.

## Honest framing

- Co-folding models are near docking or better on pose, but published PoseBusters numbers are on full sets with leakage caveats. Sequence similarity to training data matters, and the filtered subset above is an easier slice than the full set.
- The TYK2 and CDK2 series are public and in ChEMBL, which the affinity head was trained on: read the comparison as "competitive on well-known series", not as evidence about novel chemistry.
- On affinity, FEP+ is the commercial accuracy reference. The open model is a fast triage tool, not a replacement for lead optimisation.
- Cofactors and metals aren't modelled; where they shape the pocket, say so.

## Quantum: the short answer

**Research-stage test only.** Pose prediction and affinity ranking need no quantum computer, and none will help them near term. The credible link is the electronic structure of metal-containing or otherwise strongly correlated active sites, which PBE and force fields describe poorly. There the route is the same as in materials-stack: a small active space around the site, sample-based quantum diagonalization (SQD) on IBM Heron (needs the user's IBM token via vault-credentials), scored against classical CASCI or DMRG on the same active space first. It is a research experiment, not a drug-design tool. Price it with `qbraid devices get` and confirm with the user.

For the full answer and the experiment protocol, use the **quantum-readiness** skill (section "Chemistry and materials").

## Verification stamp

- 2026-10-01, `gpu-l4` instance (NVIDIA L4 24 GB, driver 595, about 5-CPU cgroup).
- Python 3.12, boltz 2.2.1, torch 2.14.1+cu130, posebusters 0.6.5, rdkit, gemmi.
- 171 GPU-minutes in 5 jobs, about $1.40; MSAs prepared on CPU via the public ColabFold server.
