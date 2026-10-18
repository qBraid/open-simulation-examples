"""AQAffinity (OpenFold3 + affinity head) queries for the FEP+ JACS ligands.

Same 2 x 16 ligands and SMILES as the Boltz-2 run (prep_affinity.py). One
difference, forced by AQAffinity (exactly one protein chain per query): the
CDK2 input is the CDK2 chain alone, without cyclin A, which Boltz-2 was given.
"""
import csv, json, sys
import gemmi
from rdkit import Chem

FEP, OUT = sys.argv[1:3]
q = {}
for t in ("tyk2", "cdk2"):
    st = gemmi.read_structure(f"{FEP}/{t}_protein.pdb"); st.setup_entities(); st.remove_ligands_and_waters()
    chains = []
    for ch in st[0]:
        res = [r for r in ch if (tr := gemmi.find_tabulated_residue(r.name)) and tr.is_amino_acid()]
        if len(res) >= 30:
            chains.append("".join((gemmi.find_tabulated_residue(r.name).one_letter_code.upper() or "G").replace("X", "G") for r in res))
    seq = chains[0]
    names = {r["Ligand name"] for r in csv.DictReader(open(f"{FEP}/{t}_out.csv"))}
    for m in Chem.SDMolSupplier(f"{FEP}/{t}_ligands.sdf", removeHs=False):
        n = m.GetProp("_Name")
        if n not in names: continue
        q[f"{t}__{n}"] = {"chains": [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": seq},
                                     {"molecule_type": "ligand", "chain_ids": ["B"], "smiles": Chem.MolToSmiles(Chem.RemoveHs(m))}]}
    print(t, "chains", [len(c) for c in chains], "using", len(seq))
json.dump({"queries": q}, open(OUT, "w"), indent=1)
print("queries", len(q))
