"""Build Boltz inputs for a reproducible PoseBusters (V1, 428) subset.

Subset rule (fixed before any prediction was made):
  take the PoseBusters V1 ids, keep complexes whose protein has <= 700
  observed residues in total and whose ligand has <= 50 heavy atoms, then
  draw 32 with random.Random(2026). Proteins are given as their observed
  residues; cofactors and other ligands are not modelled (stated limit).
MSAs come from the public ColabFold MMseqs2 server and are written as .a3m,
so the GPU step never waits on the network.
"""
import json, os, random, sys
import gemmi
from rdkit import Chem

PB = sys.argv[1]            # .../posebusters_benchmark_set
OUT = sys.argv[2]           # output dir
N, SEED, MAX_RES, MAX_HEAVY = 32, 2026, 700, 50
os.makedirs(f"{OUT}/yaml", exist_ok=True); os.makedirs(f"{OUT}/msa", exist_ok=True)

def chains(pdb):
    st = gemmi.read_structure(pdb); st.setup_entities(); st.remove_ligands_and_waters()
    out = []
    for ch in st[0]:
        res = [r for r in ch if gemmi.find_tabulated_residue(r.name) and gemmi.find_tabulated_residue(r.name).is_amino_acid()]
        if len(res) < 10: continue
        seq = "".join(gemmi.find_tabulated_residue(r.name).one_letter_code.upper() or "X" for r in res)
        out.append((ch.name, seq.replace("X", "G")))
    return out

ids = sorted(os.listdir(PB)); keep = []
for i in ids:
    lig = Chem.MolFromMolFile(f"{PB}/{i}/{i}_ligand.sdf")
    if lig is None: continue
    ch = chains(f"{PB}/{i}/{i}_protein.pdb")
    nres = sum(len(s) for _, s in ch)
    if ch and nres <= MAX_RES and lig.GetNumHeavyAtoms() <= MAX_HEAVY:
        keep.append((i, ch, Chem.MolToSmiles(lig), nres, lig.GetNumHeavyAtoms()))
print(f"eligible {len(keep)} of {len(ids)}")
sel = sorted(random.Random(SEED).sample(keep, N))

from boltz.data.msa.mmseqs2 import run_mmseqs2
meta = []
for i, ch, smi, nres, nheavy in sel:
    lines = ["version: 1", "sequences:"]
    seen = {}  # Boltz requires identical chains (homo-multimers) to share one MSA file
    for k, (cid, seq) in enumerate(ch):
        a3m = seen.setdefault(seq, f"{OUT}/msa/{i}_{k}.a3m")
        if not os.path.exists(a3m):
            res = run_mmseqs2([seq], prefix=f"{OUT}/msa/tmp_{i}_{k}", use_env=True, use_filter=True)
            open(a3m, "w").write(res[0] if isinstance(res, list) else res[0][0])
        lines += [f"  - protein:", f"      id: {chr(65+k)}", f"      sequence: {seq}", f"      msa: {os.path.abspath(a3m)}"]
    lines += ["  - ligand:", "      id: L", f"      smiles: '{smi}'"]
    open(f"{OUT}/yaml/{i}.yaml", "w").write("\n".join(lines) + "\n")
    meta.append(dict(id=i, chains=[c for c, _ in ch], n_res=nres, n_heavy=nheavy, smiles=smi))
    print("prepared", i, nres, nheavy, flush=True)
json.dump(dict(rule=__doc__, seed=SEED, n=N, eligible=len(keep), total=len(ids), complexes=meta), open(f"{OUT}/subset.json", "w"), indent=1)
