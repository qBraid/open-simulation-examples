"""Matbench Discovery protocol on a stated random sample of WBM unique prototypes.

Relax each unrelaxed WBM structure with a MACE foundation model (FIRE, fmax 0.05 eV/A,
500 steps, FrechetCellFilter: the leaderboard settings), then convert the model energy
to an MP2020-corrected formation energy. The MP2020 correction is additive per atom,
so e_form_pred = e_form_dft + (E_model - E_dft_uncorrected) / n_atoms, exactly as the
leaderboard pipeline does it. Hull distance: e_hull_pred = e_hull_dft + (e_form_pred - e_form_dft).

usage: discovery.py <model: mp0|mpa0|orb3> <n_sample> <shard> <n_shards> <out.csv> [dtype]
"""
import sys, time, zipfile, io
import numpy as np, pandas as pd
from ase.io import read
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from mace.calculators import mace_mp

model, n_sample, shard, n_shards, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
dtype = sys.argv[6] if len(sys.argv) > 6 else "float32"
SEED = 20261001
s = pd.read_csv("data/wbm-summary.csv.gz")
ids = s[s.unique_prototype].material_id.sample(n=n_sample, random_state=SEED).tolist()
ids = ids[shard::n_shards]
if model == "orb3":  # ORB v3 conservative-inf-mpa (Apache-2.0); leaderboard used fmax 0.02
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator
    orbff, adapter = pretrained.orb_v3_conservative_inf_mpa(device="cuda", precision="float32-high", compile=False)
    calc = ORBCalculator(orbff, adapter, device="cuda")  # same 2025-04-04 checkpoint as the leaderboard run
else:
    calc = mace_mp(model={"mp0": "medium", "mpa0": "medium-mpa-0"}[model], device="cuda", default_dtype=dtype)
FMAX = 0.02 if model == "orb3" else 0.05
zf = zipfile.ZipFile("data/wbm-initial-atoms.extxyz.zip")
rows = []
for i, mid in enumerate(ids):
    t0 = time.time()
    try:
        atoms = read(io.StringIO(zf.read(f"{mid}.extxyz").decode()), format="extxyz")
        atoms.calc = calc
        opt = FIRE(FrechetCellFilter(atoms), logfile=None)
        opt.run(fmax=FMAX, steps=500)
        rows.append(dict(material_id=mid, energy=atoms.get_potential_energy(), n=len(atoms),
                         steps=opt.nsteps, sec=time.time() - t0))
    except Exception as e:
        rows.append(dict(material_id=mid, energy=np.nan, n=np.nan, steps=-1, sec=time.time() - t0, err=str(e)[:200]))
    if i % 25 == 0:
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{i+1}/{len(ids)} {mid} {rows[-1]['sec']:.1f}s", flush=True)
pd.DataFrame(rows).to_csv(out, index=False)
print("done", len(rows))
