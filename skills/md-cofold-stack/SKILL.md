---
name: md-cofold-stack
description: Protein-ligand pose prediction and affinity ranking on qBraid GPUs with open models (Boltz-2, Chai-1) plus classical anchors (docking, FEP+ reference values, OpenMM). Use for co-folding a protein with a small molecule, ranking a congeneric series, benchmarking against PoseBusters or FEP+ sets, or rendering poses in three.js. Covers install, weights caching, MSA handling, GPU memory limits, the pocket-aligned RMSD + PoseBusters scoring protocol, and honest comparison rules.
---

# Co-folding and affinity stack (Boltz-2 first)

Drafted from a verified run: `open-simulation-examples/drug-discovery`. The README carries the stamp.

## Decision rules
| Question | Default | When to switch |
|---|---|---|
| Where does this ligand bind, in what pose? | **Boltz-2** co-folding (MIT, weights public) | Chai-1 (Apache) as a second opinion; classical docking (Vina, GNINA) when the pocket is known and you need hundreds of thousands of ligands |
| Rank a congeneric series | **Boltz-2 affinity head** for triage (a few minutes per ligand on an L4) | Relative FEP (OpenFE, open; FEP+, commercial) for lead optimisation. It is hours per ligand pair and still the accuracy reference |
| Is the pose physically sane? | **PoseBusters** checks against the predicted protein | Short OpenMM MD to check stability |
| What will the user see? | three.js ribbons, ball-and-stick, pocket surface, pose morph (see the example viewer) | Mol* for interactive inspection |

Always report against a classical anchor: docking success on the same complexes (the PoseBusters paper releases per-complex Vina and GOLD results), and FEP+ predictions on the same ligands (Schrödinger's public benchmark, MIT). Without that, a co-folding number means nothing.

## Operational truth on qBraid
- **Install:** `pip install boltz==2.2.1 posebusters` in a venv on local disk. It pulls torch with CUDA 13; the L4 driver 595 is fine. The venv is about 6 GB, and nearly all of that is torch and CUDA wheels.
- **No cuEquivariance kernels are installed,** so pass `--no_kernels` or Boltz fails at the triangle-attention import.
- **Weights:** about 4.3 GB (`boltz2_conf.ckpt` 2.3 GB plus `boltz2_aff.ckpt` 2.1 GB). Download them once to a large disk and symlink them into the `--cache` dir.
- **Don't let Boltz extract the full CCD `mols.tar` onto a network filesystem.** It is 45k small files and crawls. Extract only the canonical residues (`boltz.data.const.tokens` plus UNK) into a local cache dir and symlink `mols.tar` alongside. Ligands given as SMILES don't need CCD entries.
- **Generate MSAs on CPU first** with `boltz.data.msa.mmseqs2.run_mmseqs2([seq], prefix)` against the public ColabFold server (about 2 s per chain), and write `.a3m` files referenced from the YAML. Otherwise `--use_msa_server` holds a GPU slot while it waits on the network.
- **L4 (24 GB) memory:** keep complexes at 700 observed residues or fewer, with 5 diffusion samples. Multi-chain targets (for example CDK2/cyclin A in the FEP+ set) need one `protein:` entry and MSA per chain.
- **Batch GPU work** in chunks that fit the shared queue's job cap. One YAML directory per batch, with outputs per batch.
- **Use `--use_potentials`** (physics steering): it targets exactly the PoseBusters validity failures.

## Scoring protocol (do not improvise)
1. Superpose the predicted protein onto the crystal using CA atoms of residues within 10 Å of the crystal ligand.
2. Compute symmetry-aware ligand heavy-atom RMSD with RDKit `CalcRMS`, with no refitting. Assign bond orders from the input SMILES first (`AssignBondOrdersFromTemplate`).
3. Run PoseBusters `redock` checks with the predicted ligand, the crystal ligand, and the superposed predicted protein.
4. **success = RMSD ≤ 2 Å and PB-valid.** Report RMSD-only too, because published co-folding numbers are often RMSD-only.
5. Fix the subset rule and seed before predicting. Report the binomial error (±8 points at n=32).

For affinity: Spearman and Kendall with bootstrap CIs, plus a mean-centred RMSE. Boltz-2's `affinity_pred_value` is log10(IC50/µM); dG ≈ 1.364·(v−6) kcal/mol.

## Honest framing
- Co-folding models are near docking or better on pose, but published PoseBusters numbers are on full sets with leakage caveats. Sequence similarity to training data matters.
- On affinity, FEP+ is the commercial accuracy reference. The open model is a fast triage tool, not a replacement for lead optimisation.
- Quantum today: none of this needs quantum. The credible future link is fault-tolerant phase estimation for strongly correlated metal sites (see the pharma router plan in the skills design record).
