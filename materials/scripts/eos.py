"""Equilibrium lattice constant and bulk modulus for the 24-solid set of Csonka et al.,
PRB 79, 155107 (2009), arXiv:0903.4037, Tables II and V: ZPAE-corrected experiment and PBE (BAND).

Primitive cells, 11 isotropic strains within +/-6 % volume around the PBE lattice constant,
Birch-Murnaghan fit. usage: eos.py <mp0|mpa0> <out.csv>
"""
import sys
import numpy as np, pandas as pd
from ase.build import bulk
from ase.eos import EquationOfState
from ase.units import GPa
from mace.calculators import mace_mp

model, out = sys.argv[1], sys.argv[2]
calc = mace_mp(model={"mp0": "medium", "mpa0": "medium-mpa-0"}[model], device=__import__("os").environ.get("DEVICE", "cuda"), default_dtype="float64")
ref = pd.read_csv("data/csonka2009_sol24.csv")
rows = []
for r in ref.itertuples():
    a_ref = r.a0_pbe
    vols, ens = [], []
    for x in np.linspace(0.94, 1.06, 11) ** (1 / 3):
        at = bulk(r.solid, r.structure, a=a_ref * x)
        at.calc = calc
        vols.append(at.get_volume()); ens.append(at.get_potential_energy())
    eos = EquationOfState(vols, ens, eos="birchmurnaghan")
    v0, e0, B = eos.fit()
    at0 = bulk(r.solid, r.structure, a=a_ref)
    a0 = a_ref * (v0 / at0.get_volume()) ** (1 / 3)
    rows.append(dict(solid=r.solid, structure=r.structure, a0_model=a0, B0_model=B / GPa))
    print(r.solid, round(a0, 4), round(B / GPa, 1), flush=True)
pd.DataFrame(rows).merge(ref, on=["solid", "structure"]).to_csv(out, index=False)
