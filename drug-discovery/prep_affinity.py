"""Boltz-2 affinity inputs for the Schrodinger FEP+ JACS-set series (TYK2, CDK2).

Same 16 ligands per target as the public FEP+ benchmark
(github.com/schrodinger/public_binding_free_energy_benchmark, 21_4_results),
so Boltz-2, FEP+ and experiment are compared on identical compounds.
One MSA per target (ColabFold server) is shared by all ligands.
"""
import csv, os, sys
import gemmi
from rdkit import Chem

FEP, OUT = sys.argv[1], sys.argv[2]
os.makedirs(f"{OUT}/yaml", exist_ok=True); os.makedirs(f"{OUT}/msa", exist_ok=True)
from boltz.data.msa.mmseqs2 import run_mmseqs2

for t in ("tyk2", "cdk2"):
    st = gemmi.read_structure(f"{FEP}/{t}_protein.pdb"); st.setup_entities(); st.remove_ligands_and_waters()
    chains = []
    for ch in st[0]:
        res = [r for r in ch if (tr := gemmi.find_tabulated_residue(r.name)) and tr.is_amino_acid()]
        if len(res) >= 30:
            chains.append("".join((gemmi.find_tabulated_residue(r.name).one_letter_code.upper() or "G").replace("X", "G") for r in res))
    prot = []
    for k, seq in enumerate(chains):   # CDK2 is the CDK2/cyclin A complex: every chain gets its own MSA
        a3m = f"{OUT}/msa/{t}_{k}.a3m"
        if not os.path.exists(a3m):
            open(a3m, "w").write(run_mmseqs2([seq], prefix=f"{OUT}/msa/tmp_{t}_{k}", use_env=True, use_filter=True)[0])
        prot += ["  - protein:", f"      id: {chr(65+k)}", f"      sequence: {seq}", f"      msa: {os.path.abspath(a3m)}"]
    names = {r["Ligand name"] for r in csv.DictReader(open(f"{FEP}/{t}_out.csv"))}
    n = 0
    for m in Chem.SDMolSupplier(f"{FEP}/{t}_ligands.sdf", removeHs=False):
        name = m.GetProp("_Name")
        if name not in names: continue
        smi = Chem.MolToSmiles(Chem.RemoveHs(m))
        open(f"{OUT}/yaml/{t}__{name}.yaml", "w").write("\n".join([
            "version: 1", "sequences:", *prot,
            "  - ligand:", "      id: L", f"      smiles: '{smi}'", "properties:", "  - affinity:", "      binder: L", ""]))
        n += 1
    print(t, "ligands", n, "of", len(names), "chain lengths", [len(c) for c in chains], flush=True)
