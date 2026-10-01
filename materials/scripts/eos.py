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
import os
if model == "orb3":  # ORB v3 conservative-inf-mpa (Apache-2.0)
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator
    _m, _ad = pretrained.orb_v3_conservative_inf_mpa(device=os.environ.get("DEVICE", "cuda"), precision="float64", compile=False)
    calc = ORBCalculator(_m, _ad, device=os.environ.get("DEVICE", "cuda"))
else:
    calc = mace_mp(model={"mp0": "medium", "mpa0": "medium-mpa-0", "mp0b3": "medium-0b3"}[model], device=os.environ.get("DEVICE", "cuda"), default_dtype="float64")
ref = pd.read_csv("data/csonka2009_sol24.csv")
rows = []
for r in ref.itertuples():
    def scan(a_c, lo, hi, n):
        vols, ens = [], []
        for x in np.linspace(lo, hi, n) ** (1 / 3):
            at = bulk(r.solid, r.structure, a=a_c * x); at.calc = calc
            vols.append(at.get_volume()); ens.append(at.get_potential_energy())
        return np.array(vols), np.array(ens)
    a_ref, how = r.a0_pbe, "bm"
    vols, ens = scan(a_ref, 0.94, 1.06, 11)
    try:
        v0, e0, B = EquationOfState(vols, ens, eos="birchmurnaghan").fit()
        if not (vols.min() < v0 < vols.max()):
            raise RuntimeError("minimum outside window")
    except Exception:
        # recentre: coarse scan over -25%..+35% volume, then refit around the minimum
        cv, ce = scan(a_ref, 0.75, 1.35, 25)
        vmin = cv[np.argmin(ce)]
        a_ref = a_ref * (vmin / bulk(r.solid, r.structure, a=a_ref).get_volume()) ** (1 / 3)
        vols, ens = scan(a_ref, 0.94, 1.06, 11)
        try:
            v0, e0, B = EquationOfState(vols, ens, eos="birchmurnaghan").fit(); how = "bm-recentred"
        except Exception:
            v0, e0, B = EquationOfState(vols, ens, eos="sj").fit(); how = "sjeos-recentred"
    at0 = bulk(r.solid, r.structure, a=a_ref)
    a0 = a_ref * (v0 / at0.get_volume()) ** (1 / 3)
    rows.append(dict(solid=r.solid, structure=r.structure, a0_model=a0, B0_model=B / GPa, fit=how))
    print(r.solid, round(a0, 4), round(B / GPa, 1), flush=True)
pd.DataFrame(rows).merge(ref, on=["solid", "structure"]).to_csv(out, index=False)
