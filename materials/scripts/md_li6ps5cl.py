"""Li-ion diffusion in the argyrodite solid electrolyte Li6PS5Cl (mp-985592, ordered) with MACE MD.

2x2x2 conventional supercell (416 atoms), cell relaxed with the model, then NVT Langevin at T
for `ps` picoseconds (2 fs step, first 20 % discarded). Li tracer diffusivity from the MSD slope.
Saves a thinned trajectory for the viewer. usage: md_li6ps5cl.py <mp0|mpa0> <T_K> <ps> <out_prefix>
"""
import sys, json, time
import numpy as np
from ase import Atoms, units
from ase.build import make_supercell
from ase.filters import FrechetCellFilter
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import FIRE
from mace.calculators import mace_mp

model, T, ps, pre = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
d = json.load(open("data/mp-985592_Li6PS5Cl.json"))
prim = Atoms(d["species_at_sites"], positions=d["cartesian_site_positions"], cell=d["lattice_vectors"], pbc=True)
P = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]) * 2  # fcc primitive -> 2x2x2 conventional
atoms = make_supercell(prim, P)
import os
calc = mace_mp(model={"mp0": "medium", "mpa0": "medium-mpa-0", "mp0b3": "medium-0b3"}[model], device=os.environ.get("DEVICE", "cuda"), default_dtype="float32")
atoms.calc = calc
FIRE(FrechetCellFilter(atoms), logfile=None).run(fmax=0.05, steps=300)
MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=np.random.default_rng(7))
dyn = Langevin(atoms, 2 * units.fs, temperature_K=T, friction=0.01 / units.fs, rng=np.random.default_rng(11))
li = np.array([s == "Li" for s in atoms.get_chemical_symbols()])
nsteps = int(ps * 500); every = 25  # frame every 50 fs
frames, unwrapped, prev, shift = [], [], atoms.get_scaled_positions(), np.zeros((len(atoms), 3))
t0 = time.time()
for k in range(nsteps // every):
    dyn.run(every)
    sp = atoms.get_scaled_positions()
    shift -= np.round(sp - prev); prev = sp
    unwrapped.append(((sp + shift) @ atoms.cell.array).astype(np.float32))
    frames.append(atoms.get_positions().astype(np.float32))
el = time.time() - t0
U = np.array(unwrapped); n0 = len(U) // 5
lags = np.arange(1, (len(U) - n0) // 2)
msd = np.array([((U[n0 + l:, li] - U[n0:-l, li]) ** 2).sum(-1).mean() for l in lags])
dt_ps = every * 2e-3
slope = np.polyfit(lags * dt_ps, msd, 1)[0]  # A^2/ps
D = slope / 6 * 1e-4  # cm^2/s  (1 A^2/ps = 1e-4 cm^2/s)
res = dict(model=model, T=T, ps=ps, n_atoms=len(atoms), a_conv=float(atoms.cell.lengths()[0] / 2),
           D_Li_cm2_s=float(D), steps_per_s=nsteps / el, msd_lag_ps=(lags * dt_ps).tolist(), msd_A2=msd.tolist())
json.dump(res, open(pre + ".json", "w"))
np.savez_compressed(pre + "_traj.npz", frames=np.array(frames), symbols=np.array(atoms.get_chemical_symbols()), cell=atoms.cell.array)
print(json.dumps({k: v for k, v in res.items() if not k.startswith("msd")}))
