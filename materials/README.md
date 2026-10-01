# Materials Discovery Lab

Universal machine-learned interatomic potentials (MLIPs) on a single qBraid L4 GPU. They are benchmarked on three things:
- the community leaderboard for discovering stable crystals (Matbench Discovery);
- experiment and PBE, for lattice constants, bulk moduli and phonons;
- a lithium-ion MD showcase in a battery solid electrolyte.

Every model used here has a commercially usable licence (MIT or Apache-2.0).

Open `viewer.html` (or `qbraid-canvas materials/viewer.html`) for the interactive version. It has four scenes and a guided tour.

<!--RESULTS-->

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
- **Sample, not the full test set.** We ran 2,000 of 215,488 structures. The bootstrap 95% interval on F1 is about ±0.05. Exactness is shown instead by per-structure agreement with the authors' own predictions.
- **The top-10% models were not run.** EquiformerV3+DeNS-OAM (MIT, F1 0.931) needs its own research build of fairchem (torch 2.4, numpy < 2, about 5 GB). The full leaderboard run took 48 H200-hours, so a 500-structure check would take about 40 minutes on an L4, or about 7 minutes on an H100 (roughly $0.60).
- **PBE is the ceiling.** For lattice constants these models cannot beat PBE's own error against experiment of about 1.4% (overestimate). VASP with a better functional (PBEsol, r²SCAN) would do better. That is the place to spend DFT time, after MLIP screening.
- **Cutoff artefacts.** Cs (and partly Rb and Na) has neighbour shells near the 6 Å graph cutoff, so E(V) curves have kinks. Cs is excluded from the EOS statistics and shown in the viewer.
- **MD uses the ordered Materials Project cell** (mp-985592) of Li₆PS₅Cl. Real samples have S/Cl site disorder, which raises conductivity. Activation energies and extrapolated room-temperature conductivity are indicative, not predictive.

## Where quantum fits
These are classical surrogates of PBE. Quantum computing is relevant where PBE itself fails: strongly correlated transition-metal and rare-earth sites in catalysts, magnets and some battery cathodes. The route there is the chemistry stack: an active-space Hamiltonian, then VQE or SQD on today's hardware (at sizes still classically simulable, so a research and readiness exercise), and fault-tolerant phase estimation in the future. The practical hybrid today is to screen with an MLIP, confirm with DFT, and flag the strongly correlated candidates for active-space treatment.
