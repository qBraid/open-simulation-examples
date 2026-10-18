"""OpenFold3 ligand coordinates for the viewer's showcase complexes, in the crystal frame.

eval_poses.py already superposed each prediction onto the crystal pocket and wrote
pred_ligand.sdf. Atoms are re-ordered to the crystal ligand's order and centred on
the crystal ligand centroid, the same frame drug-discovery/viewer.html uses.
usage: of3_showcase.py <posebusters_set_dir> <of3_eval_dir> <out.json> <id> [<id> ...]
"""
import json, sys
import numpy as np
from rdkit import Chem

PB, EV, OUT, *IDS = sys.argv[1:]
out = {}
for i in IDS:
    ref = Chem.MolFromMolFile(f"{PB}/{i}/{i}_ligand.sdf")
    pred = Chem.MolFromMolFile(f"{EV}/{i}/pred_ligand.sdf")
    match = pred.GetSubstructMatch(ref)
    c = ref.GetConformer().GetPositions().mean(0)
    out[i] = np.round(pred.GetConformer().GetPositions()[list(match)] - c, 3).tolist() if match else None
json.dump(out, open(OUT, "w"))
print({k: (len(v) if v else None) for k, v in out.items()})
