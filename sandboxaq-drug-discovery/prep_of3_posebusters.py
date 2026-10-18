"""OpenFold3 query JSON for the same 32 PoseBusters complexes Boltz-2 was scored on.

Reads the subset fixed by prep_posebusters.py (results/posebusters_subset.json)
so the two models see identical inputs: observed protein residues per chain,
one ligand from SMILES (chain L), no cofactors. Chains keep the A, B, ... order
eval_poses.py pairs with the crystal. MSAs come from the ColabFold server at run
time; templates are off, as in the Boltz-2 run.
"""
import json, sys
import gemmi

PB, SUBSET, OUT = sys.argv[1:4]
meta = json.load(open(SUBSET))

def chains(pdb):  # identical to prep_posebusters.py
    st = gemmi.read_structure(pdb); st.setup_entities(); st.remove_ligands_and_waters()
    out = []
    for ch in st[0]:
        res = [r for r in ch if gemmi.find_tabulated_residue(r.name) and gemmi.find_tabulated_residue(r.name).is_amino_acid()]
        if len(res) < 10: continue
        out.append("".join(gemmi.find_tabulated_residue(r.name).one_letter_code.upper() or "X" for r in res).replace("X", "G"))
    return out

q = {}
for c in meta["complexes"]:
    i = c["id"]
    seqs = chains(f"{PB}/{i}/{i}_protein.pdb")
    q[i] = {"chains": [{"molecule_type": "protein", "chain_ids": [chr(65 + k)], "sequence": s} for k, s in enumerate(seqs)]
                      + [{"molecule_type": "ligand", "chain_ids": ["L"], "smiles": c["smiles"]}]}
json.dump({"queries": q}, open(OUT, "w"), indent=1)
print("queries", len(q))
