"""Score Boltz co-folding predictions on the PoseBusters subset.

Protocol (as in the AlphaFold3 / PoseBusters co-folding evaluations):
  1. superpose the predicted protein onto the crystal using CA atoms of pocket
     residues (crystal residues with any atom within 10 A of the ligand);
  2. ligand heavy-atom RMSD, symmetry-aware, no further fitting;
  3. PoseBusters "redock" checks of the predicted ligand against the
     predicted (superposed) protein. success = RMSD <= 2 A and PB-valid.
Classical baselines on the same complexes come from the PoseBusters paper's
released per-complex results (Vina, GOLD: crystal protein, known pocket).
"""
import csv, glob, json, os, sys, warnings
import numpy as np, gemmi
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign
RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore")

PB, PRED, META, PBRES, OUT = sys.argv[1:6]
meta = json.load(open(META))

def aa_residues(st):
    out = []
    for ch in st[0]:
        res = [r for r in ch if (t := gemmi.find_tabulated_residue(r.name)) and t.is_amino_acid()]
        if len(res) >= 10: out.append(res)
    return out

def kabsch(P, Q):  # find R,t minimising |R P + t - Q|
    pc, qc = P.mean(0), Q.mean(0); H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T; return R, qc - R @ pc

rows = []
for c in meta["complexes"]:
    i = c["id"]; row = dict(id=i, n_res=c["n_res"], n_heavy=c["n_heavy"])
    pdbs = glob.glob(f"{PRED}/**/{i}_model_0.pdb", recursive=True)
    if not pdbs: row["status"] = "no prediction"; rows.append(row); continue
    xtal = gemmi.read_structure(f"{PB}/{i}/{i}_protein.pdb"); xtal.setup_entities()
    ref = Chem.MolFromMolFile(f"{PB}/{i}/{i}_ligand.sdf")
    L = ref.GetConformer().GetPositions()
    pred = gemmi.read_structure(pdbs[0])
    xr, pr = aa_residues(xtal), [ [r for r in ch if gemmi.find_tabulated_residue(r.name) and gemmi.find_tabulated_residue(r.name).is_amino_acid()] for ch in pred[0] if ch.name != "L"]
    P, Q, Pall, Qall = [], [], [], []
    for xch, pch in zip(xr, pr):
        for a, b in zip(xch, pch):
            if a.find_atom("CA", "*") is None or b.find_atom("CA", "*") is None: continue
            x = np.array(a.find_atom("CA", "*").pos.tolist()); y = np.array(b.find_atom("CA", "*").pos.tolist())
            Qall.append(x); Pall.append(y)
            if min(np.linalg.norm(np.array(at.pos.tolist()) - L, axis=1).min() for at in a) <= 10.0:
                Q.append(x); P.append(y)
    R, t = kabsch(np.array(P), np.array(Q))
    row["pocket_ca"] = len(P)
    row["pocket_ca_rmsd"] = float(np.sqrt((((np.array(P) @ R.T + t) - np.array(Q)) ** 2).sum(1).mean()))
    # transform the whole prediction into the crystal frame
    for ch in pred[0]:
        for r in ch:
            for at in r:
                v = R @ np.array(at.pos.tolist()) + t; at.pos = gemmi.Position(*v)
    os.makedirs(f"{OUT}/{i}", exist_ok=True)
    lig_lines, prot = [], gemmi.Structure(); prot.add_model(gemmi.Model("1"))
    for ch in pred[0]:
        if ch.name == "L": continue
        prot[0].add_chain(ch)
    prot.setup_entities(); prot.write_pdb(f"{OUT}/{i}/pred_protein.pdb")
    pred.setup_entities()
    block = "\n".join(l for l in pred.make_pdb_string().splitlines() if l.startswith("HETATM") and l[21] == "L") + "\nEND\n"
    pl = Chem.MolFromPDBBlock(block, removeHs=False, proximityBonding=True)
    try:
        pl = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(c["smiles"]), Chem.RemoveHs(pl, sanitize=False))
        Chem.SanitizeMol(pl)
    except Exception as e:
        row["status"] = f"bond-order assignment failed: {e}"[:80]; rows.append(row); continue
    Chem.MolToMolFile(pl, f"{OUT}/{i}/pred_ligand.sdf")
    row["rmsd"] = float(rdMolAlign.CalcRMS(pl, ref))
    conf = glob.glob(f"{os.path.dirname(pdbs[0])}/confidence_{i}_model_0.json")
    if conf:
        cj = json.load(open(conf[0])); row.update(confidence=cj.get("confidence_score"), ligand_iptm=cj.get("ligand_iptm"), iptm=cj.get("iptm"))
    try:
        from posebusters import PoseBusters
        df = PoseBusters(config="redock").bust(mol_pred=f"{OUT}/{i}/pred_ligand.sdf", mol_true=f"{PB}/{i}/{i}_ligand.sdf", mol_cond=f"{OUT}/{i}/pred_protein.pdb")
        checks = {k: bool(v) for k, v in df.iloc[0].items() if isinstance(v, (bool, np.bool_))}
        row["pb_failed"] = [k for k, v in checks.items() if not v and "rmsd" not in k]
        row["pb_n_checks"] = sum("rmsd" not in k for k in checks)
        row["pb_valid"] = len(row["pb_failed"]) == 0
    except Exception as e:
        row["pb_valid"] = None; row["pb_error"] = str(e)[:120]
    row["success"] = bool(row["rmsd"] <= 2.0 and row.get("pb_valid"))
    row["status"] = "ok"; rows.append(row); print(i, f"rmsd={row['rmsd']:.2f}", "pb_valid", row.get("pb_valid"), flush=True)

# classical baselines on exactly these complexes (PoseBusters paper results)
base = {}
for r in csv.DictReader(open(PBRES)):
    if r["dataset"] != "posebuster" or r["method"] not in ("vina", "gold") or r["post-processing"] != "none": continue
    key = f"{r['pdb_id']}_{r['ccd_id']}"
    checks = [k for k in r if k not in ("dataset", "method", "post-processing", "pdb_id", "ccd_id", "has_cofactors", "sequence_identity", "rmsd", "rmsd_within_threshold")]
    valid = all(r[k] == "True" for k in checks)
    ok = r["rmsd_within_threshold"] == "True"
    base.setdefault((r["method"], r["post-processing"]), {})[key] = dict(rmsd_ok=ok, success=ok and valid)
ids = [r["id"] for r in rows]
ok_rows = [r for r in rows if r.get("status") == "ok"]
summary = dict(n=len(rows), n_scored=len(ok_rows),
    boltz2=dict(rmsd_le_2=sum(r["rmsd"] <= 2 for r in ok_rows) / len(rows), success_pb_valid=sum(r["success"] for r in ok_rows) / len(rows)))
for (m, pp), d in base.items():
    hit = [d[i] for i in ids if i in d]
    if hit: summary[m] = dict(n=len(hit), rmsd_le_2=sum(h["rmsd_ok"] for h in hit) / len(hit), success_pb_valid=sum(h["success"] for h in hit) / len(hit))
json.dump(dict(summary=summary, per_complex=rows), open(f"{OUT}/poses.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
