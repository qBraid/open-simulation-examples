# SandboxAQ drug-discovery models on qBraid

SandboxAQ's drug-discovery models, run from qBraid and scored on the same benchmark as the open baseline in [`drug-discovery/`](../drug-discovery/). The open models were OpenFold3 (pose prediction) and AQAffinity (binding affinity), both on a qBraid GPU. The hosted model was AQPotency, called from Claude Code through SandboxAQ's MCP server. All three use the same 32 PoseBusters complexes and the same 32 TYK2 and CDK2 ligands as Boltz-2 and Schrödinger's FEP+. The interactive three.js viewer is `viewer.html`.

**Why this example exists:** SandboxAQ ships its Large Quantitative Models in two ways. The open ones are on Hugging Face (OpenFold3, AQAffinity, AQCat25, AQVolt26). The hosted ones are behind an MCP server that AI agents call (AQPotency, AQCat, AQAffinity). This example checks that both ways work end to end on qBraid, and measures what each model adds over the open baseline.

## What was run

| Model | From SandboxAQ | Access | Licence | Where it ran |
|---|---|---|---|---|
| **OpenFold3** (preview, `openbind-2025-06-30` weights) | Co-developed in the OpenFold Consortium; SandboxAQ is a core contributor | `pip install openfold3`; weights from the public OpenFold S3 bucket | Apache-2.0 | qBraid `gpu-a100-sxm` |
| **AQAffinity** (preview) | SandboxAQ, Feb 2026: an open replication of the Boltz-2 affinity head on OpenFold3 | Hugging Face `SandboxAQ/AQAffinity` (gated: a short form, instant) | Apache-2.0 | qBraid `gpu-a100-sxm` |
| **AQPotency** (GA Aug 2026) | SandboxAQ proteochemometric model: UniProt ID plus SMILES gives pIC50 ± σ | Hosted: SandboxAQ MCP server, OAuth login, free starter credits | Commercial service | SandboxAQ's servers, called from Claude Code on qBraid |

## Results

### Pose prediction: the same 32 PoseBusters V1 complexes
| Method | RMSD ≤ 2 Å | and PB-valid |
|---|---|---|
| Boltz-2 (`drug-discovery/`, with physics steering potentials) | 84% | 84% |
| **OpenFold3 preview, this run** | **72%** | **66%** |
| GOLD docking (crystal protein and pocket given) | 41% | 34% |
| VINA docking (crystal protein and pocket given) | 50% | 44% |
| AlphaFold3, full V1 set (published) | 76% | – |

**Below Boltz-2 on this subset, and above docking.** OpenFold3 succeeds on 21 of 32 complexes and Boltz-2 on 27 of 32. Boltz-2 succeeds on every complex where OpenFold3 does. The gap is 6 complexes, about 1.5σ at n = 32, so the subset only weakly separates the two models.
- **Physics checks account for 2 of the 6.** 6ZK5_IMH (1.6 Å) and 7ULC_56B (1.3 Å) have the right pose but fail PoseBusters checks: a protein clash and wrong chirality. Boltz-2 was run with physics steering potentials, which are designed to prevent exactly that; OpenFold3 has no equivalent option.
- **The other 4 are pose misses:** 6ZR8_QOZ (2.6 Å), 8A2D_KXY (2.9 Å), 7Q19_DSM (4.2 Å) and 7BHX_TO5 (33 Å, wrong pocket, confidence 0.69).
- **Both models miss the same 5 hard cases.** These include 5SAK_ZRY and 8BRO_R7E, which are confident wrong pockets.

### Affinity ranking: Schrödinger FEP+ JACS series, identical ligands
| Target | n | FEP+ (commercial physics) | Boltz-2 (`drug-discovery/`) | **AQAffinity, OpenFold3 0.5** | AQAffinity, OpenFold3 0.3.1 | AQPotency (hosted) |
|---|---|---|---|---|---|---|
| TYK2 Spearman [95% CI] | 16 | 0.93 [0.75, 0.99] | 0.77 [0.40, 0.94] | **0.45** [−0.02, 0.78] | 0.49 [−0.10, 0.86] | 0.74 † |
| TYK2 Kendall · RMSE (centred, kcal/mol) | | 0.81 · 0.47 | 0.60 · 0.74 | 0.33 · 1.13 | 0.35 · 1.02 | 0.53 · 0.92 |
| CDK2 Spearman [95% CI] | 16 | 0.62 [0.17, 0.89] | 0.87 [0.63, 0.94] | **0.84** [0.57, 0.95] | 0.36 [−0.22, 0.79] | 0.94 † |
| CDK2 Kendall · RMSE (centred, kcal/mol) | | 0.44 · 0.90 | 0.63 · 0.69 | 0.68 · 0.76 | 0.27 · 1.19 | 0.82 · 0.42 |

† AQPotency flags 15 of 16 ligands in each series as already in its training data for that protein (`ligand_similarity` = 1.0). Its column measures recall of known data, not prediction.

**AQAffinity does not beat Boltz-2 here.** On CDK2 it ties Boltz-2 (0.84 vs 0.87), on TYK2 it falls well short (0.45 vs 0.77), and on both series its intervals overlap Boltz-2's.
- **The current OpenFold3 stack is the stronger setup.** It gives the better CDK2 ranking: 0.84, against 0.36 on the 0.3.1 stack with the `of3_ft3_v1` weights, which was current when AQAffinity was released. The head isn't tied to one trunk checkpoint, but its results move a lot with the trunk.
- **AQPotency ranks well (0.74 and 0.94), but on molecules it has seen.** It needs no structure, GPU or MSA, and returns all 16 ligands in about 5 seconds. Its own similarity flag says these series are in its training set, which is the honest reading of those numbers.
- **These are well-known 2015 series.** Every model here may have seen them, so none of the numbers is evidence about new chemistry (see Honest limits).

## Method
- **Same inputs as the baseline.** The 32 complexes are the subset fixed in `drug-discovery/results/posebusters_subset.json`. The 2 × 16 ligands are the FEP+ JACS TYK2 and CDK2 series, identical SMILES. Prep scripts: `prep_of3_posebusters.py`, `prep_aqaffinity.py`, `inputs/`.
- **OpenFold3 settings.** `openfold3` 0.5.0 with default checkpoint `openbind-2025-06-30-174k`. 5 diffusion samples, 1 model seed, ColabFold MSA server, templates off (as in the Boltz-2 run). The top sample by OpenFold3's `sample_ranking_score` is scored (`of3_to_eval.py`). Scoring is `drug-discovery/eval_poses.py`, unchanged: pocket-Cα superposition, symmetry-aware RMSD, and PoseBusters `redock` checks.
- **AQAffinity settings.** `aqaffinity predict` with the Hugging Face head weights (`model_weights_only.safetensors`), seed 16. OpenFold3 makes 5 structures and the head scores the most confident one, as Boltz-2 does. There are two runs: openfold3 0.5.0 with `openbind-2025-06-30-174k` (the current default), and openfold3 0.3.1 with `of3_ft3_v1` (current when AQAffinity was released). AQAffinity takes exactly one protein chain, so CDK2 is given without cyclin A, which Boltz-2 had (`prep_aqaffinity.py`).
- **AQAffinity's sign is the opposite of Boltz-2's,** despite the "direct replication" wording. A higher output means stronger binding, about −log10(IC50/µM). The evidence: its decoy loss penalises decoys unless the prediction is below −3, and its training labels have median +0.96. Read with Boltz-2's convention, the rankings come out negative on both targets and both stacks. It is converted with ΔG = −1.364 · (v + 6) (`eval_aqaffinity.py`).
- **AQPotency settings.** `AqpotencyScreen` on UniProt P29597 (TYK2) and P24941 (CDK2), one 16-SMILES upload per target. pIC50 is converted to ΔG = −1.364 · pIC50 (`eval_aqpotency.py`). The raw responses, job IDs included, are in `results/aqpotency_raw.json`.
- **Same metrics everywhere.** Spearman and Kendall against experiment, with 2,000-resample bootstrap 95% CIs, plus mean-centred RMSE, all identical to `drug-discovery/eval_affinity.py`.

## Reproduce
```bash
# GPU instance (A100 80 GB used; OpenFold3 needs more memory than Boltz-2)
uv venv of3 && VIRTUAL_ENV=of3 uv pip install openfold3 "biotite<1.5" "lightning-utilities<0.15" "pytorch-lightning<2.6"
# kalign2 for templates: micromamba create -p kal -c bioconda kalign2 ; export PATH=kal/bin:$PATH
python prep_of3_posebusters.py <posebusters_benchmark_set> ../drug-discovery/results/posebusters_subset.json of3_pb.json
run_openfold predict --query_json of3_pb.json --use_msa_server true --use_templates false --num_diffusion_samples 5 --num_model_seeds 1 --output_dir out_of3_pb
python of3_to_eval.py out_of3_pb of3_eval_in
python ../drug-discovery/eval_poses.py <posebusters_benchmark_set> of3_eval_in ../drug-discovery/results/posebusters_subset.json pbres.csv eval_of3
# AQAffinity: accept the form at huggingface.co/SandboxAQ/AQAffinity, download the repo, then
VIRTUAL_ENV=of3 uv pip install --no-deps ./aqaffinity safetensors
python prep_aqaffinity.py <fep_dir> aqa_fep.json
python aqaffinity_run.py predict --query_json aqa_fep.json --runner_yaml runner_aqaffinity.yaml --use_msa_server true \
  --inference_ckpt_path ~/.openfold3/of3-ob-2025-06-30-174k.pt --binding_affinity_ckpt_path aqaffinity/model_weights/model_weights_only.safetensors --output_dir out_aqa
python eval_aqaffinity.py out_aqa:aqaffinity out_aqa_old:aqaffinity_of3v031 ../drug-discovery/results ../drug-discovery/results/affinity.json results/aqaffinity.json
# AQPotency (hosted): in Claude Code,
#   claude mcp add --scope user --transport http sandboxaq-mcp https://mcp.platform.aisim.sandboxaq.com
# authenticate, then ask the agent to screen inputs/tyk2.smi against P29597 and inputs/cdk2.smi against P24941
python eval_aqpotency.py results/aqpotency_raw.json inputs/aqpotency_inputs.csv results/aqpotency.json
python build_viewer.py
```

### Setup gotchas measured on qBraid (October 2026)
- **`pip install openfold3` (0.5.0) pins nothing**, and the newest `biotite` and `lightning-utilities` break inference ("maximum supported dimension for an ndarray is 64"). Pin `biotite<1.5`, `lightning-utilities<0.15` and `pytorch-lightning<2.6`.
- **OpenFold3 weights don't need the Hugging Face gate:** `openfold3.entry_points.parameters.download_model_parameters` pulls from the public S3 bucket. Only AQAffinity is gated.
- **AQAffinity (written against openfold3 0.3) crashes on 0.5** in its second pass ("FileExistsError" on the MSA workspace). There are two fixes: `aqaffinity_run.py` allows the existing workspace, and the runner YAML sets `msa_computation_settings.msa_output_directory` explicitly (`runner_aqaffinity.yaml`).
- **AQAffinity's head hard-codes DeepSpeed's evoformer kernel,** which needs a DeepSpeed build with CUDA ops (`nvcc`) that these images lack. `aqaffinity_run.py` runs the head with OpenFold3's standard attention instead: the same computation, without the memory-saving kernel.
- **The affinity stage calls the public ColabFold server again for every ligand,** about 30 s each, so it is network-bound; the GPU sits mostly idle. For more than a few dozen ligands, run a local MSA server or cache the alignments.
- **On openfold3 0.3.1, uninstall DeepSpeed** (it needs `CUDA_HOME`) **and set** `model_update.custom.settings.memory.eval.use_deepspeed_evo_attention: false` while keeping `presets: [predict, pae_enabled]` (`runner_aqaffinity_of3v031.yaml`). Its weights are at `s3://openfold/openfold3_params/of3_ft3_v1.pt`, which is public over HTTPS.
- **The SandboxAQ MCP login is OAuth in a browser.** On a remote pod, the redirect to `localhost` fails. Paste that URL back to the agent, which completes the login; each link is single-use and expires within minutes.

## Honest limits
- **Most of the affinity test set is in the training data.** AQPotency reports `ligand_similarity = 1.0`, an exact match to a training ligand for that protein, for 15 of 16 TYK2 and 15 of 16 CDK2 ligands. One CDK2 ligand appears verbatim in AQAffinity's published training set. The series are public in ChEMBL, which Boltz-2 also trained on. These numbers say how well each model reproduces a well-known series, not how it does on new chemistry. A fair test needs a series published after every model's data cutoff.
- **32 complexes and 16 ligands per target are small samples:** about ±8 points on pose success, and wide CIs on Spearman.
- **OpenFold3 is a preview** ("final model in development", per its repository), and AQAffinity is a preview head. OpenFold3 had no physics steering, while Boltz-2 did.
- **CDK2 input differs:** AQAffinity takes one protein chain, so cyclin A was dropped; Boltz-2 had both chains.
- **Hosted results depend on SandboxAQ's servers.** AQPotency's version isn't pinned by the API. The job IDs are recorded, so the runs can be traced.

## Why use the SandboxAQ models if Boltz-2 scores higher here
On this benchmark, the open Boltz-2 (MIT) beats OpenFold3 on poses and AQAffinity on TYK2, and ties AQAffinity on CDK2. That is not the whole picture.
- **The gaps are small against the noise.** The pose gap is 6 of 32 complexes, and 2 of those 6 are right poses that fail a physics check, which post-hoc relaxation can recover. On affinity, every AQAffinity interval overlaps Boltz-2's.
- **This is open versus open.** OpenFold3 and AQAffinity are Apache-2.0 with training code, developed in a non-profit consortium. The comparison is between two open model families, not a product against the community.
- **The real use is fine-tuning on private data.** OpenFold3 is built to be retrained, and the consortium's pharma members are fine-tuning it on their own unpublished structures. For a drug company, the question is which base model it can legally and practically adapt to its targets. Out-of-the-box accuracy on public series is a weaker signal.
- **AQAffinity is a research release.** It shows that the Boltz-2 affinity method transfers to a different trunk, and the CDK2 result (0.84 vs 0.87) supports that. Its results move with the trunk version (0.36 to 0.84 on CDK2), so it should improve as OpenFold3 leaves preview.
- **The product is the hosted service.** AQPotency needs no structure or GPU and ranked best here, but on molecules it had seen. Its value on new chemistry is untested.
- **What would settle it:** a congeneric series published after every model's data cutoff, scored with this same pipeline.

## Where this fits for qBraid
- **SandboxAQ's hosted models work from a qBraid agent today.** The agent calls the MCP server directly from a qBraid pod. Our 32-ligand screen cost about $0.21 of free credit and took seconds.
- **The open weights install and run on qBraid GPUs**, with the version pins above. They can be packaged as a qBraid environment, so a user doesn't repeat the setup.
- **Fine-tuning on private data.** OpenFold3 can be retrained on a customer's own structures on qBraid GPUs, then scored with this benchmark against Boltz-2 and FEP+.
- **The usability fixes are upstreamable.** The fixes for the sign convention, the openfold3 0.5 crash, the DeepSpeed dependency and the per-ligand MSA calls (see Method and Setup gotchas) are concrete fixes to offer SandboxAQ.
- **A natural funnel.** AQPotency triages a library cheaply on CPU-backed hosted calls. Co-folding (Boltz-2 or OpenFold3 + AQAffinity) then checks the top hits on qBraid GPUs. Physics (FEP, or quantum chemistry via SandboxAQ's Tangelo on qBraid QPUs) runs on the last few.

## Verification stamp
- date: 2026-10-16
- machine: qBraid gpu-a100-sxm on-demand instance (NVIDIA A100-SXM4 80 GB, driver 580.159.04), terminated after the run
- env: openfold3 0.5.0 (torch 2.14.1, biotite 1.4.0, pytorch-lightning 2.5.6) and openfold3 0.3.1 (torch 2.10.0, biotite 1.2.0) venvs; aqaffinity 0.0.1 from Hugging Face; posebusters 0.6.5, rdkit
- gpu_minutes_billed: 396
- credits_billed: 1476.72
- cost_usd: 14.77
- compute_note: About 2 h of the 6.6 h billed was model compute (OpenFold3 PoseBusters 12 min; each AQAffinity run about 45-90 min, dominated by ColabFold MSA server calls). The rest was setup, failed attempts fixed above, and idle time after the subscription pod was stopped mid-run. A clean rerun is about 2 A100-hours, about 450 credits.
- aqpotency: 2 AqpotencyScreen jobs of 16 ligands on SandboxAQ's servers, 4-6 s each, about $0.21 of SandboxAQ starter credit (their price list: $0.01 per pair for the first 10, then $0.001)
- viewer: built on the subscription pod at no extra cost
