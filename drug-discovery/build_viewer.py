"""Build viewer.html: inlines pose, pocket, affinity and benchmark data into viewer_template.html.

usage: build_viewer.py <posebusters_set_dir> <pose_eval_dir> <results_dir>
Needs: rdkit, gemmi, numpy, scikit-image, scikit-learn.
"""
import json, os, sys
import numpy as np, gemmi
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from skimage import measure
RDLogger.DisableLog("rdApp.*")
PB, EV, RES = sys.argv[1:4]
HERE = os.path.dirname(os.path.abspath(__file__))

def ss_assign(ca):
    """CA-only secondary structure with P-SEA distance windows (Labesse et al. 1997):
    helix  d(i,i+2)=5.5+-0.5, d(i,i+3)=5.3+-0.5, d(i,i+4)=6.4+-0.6
    strand d(i,i+2)=6.7+-0.6, d(i,i+3)=9.9+-0.9, d(i,i+4)=12.4+-1.1
    a residue run must reach 5 (helix) or 3 (strand) consecutive hits."""
    n = len(ca); s = ["C"] * n
    d = lambda i, j: float(np.linalg.norm(ca[i] - ca[j])) if j < n else 99.0
    hel = [abs(d(i, i+2) - 5.5) < .5 and abs(d(i, i+3) - 5.3) < .5 and abs(d(i, i+4) - 6.4) < .6 for i in range(n)]
    strd = [abs(d(i, i+2) - 6.7) < .6 and abs(d(i, i+3) - 9.9) < .9 and abs(d(i, i+4) - 12.4) < 1.1 for i in range(n)]
    for flags, tag, run in ((hel, "H", 5), (strd, "E", 3)):
        i = 0
        while i < n:
            j = i
            while j < n and flags[j]: j += 1
            if j - i >= run:
                for k in range(i, min(n, j + (4 if tag == "H" else 2))):
                    if s[k] == "C": s[k] = tag
            i = max(j, i + 1)
    return s

def segments(ca, ss):
    segs, start = [], 0
    for i in range(1, len(ca) + 1):
        if i == len(ca) or ss[i] != ss[start]:
            pts = ca[max(0, start - 1): min(len(ca), i + 1)]
            if len(pts) >= 2: segs.append(dict(t=ss[start], p=np.round(pts, 2).tolist()))
            start = i
    return segs

def mol_json(m, center):
    pos = m.GetConformer().GetPositions() - center
    return dict(a=[[a.GetSymbol(), *np.round(p, 3).tolist()] for a, p in zip(m.GetAtoms(), pos)],
                b=[[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble()] for b in m.GetBonds()])

HYD = set("ALA VAL LEU ILE MET PHE TRP PRO CYS GLY TYR".split()); POS = {"LYS", "ARG", "HIS"}; NEG = {"ASP", "GLU"}

def complex_data(i, row):
    ref = Chem.MolFromMolFile(f"{PB}/{i}/{i}_ligand.sdf")
    center = ref.GetConformer().GetPositions().mean(0)
    st = gemmi.read_structure(f"{PB}/{i}/{i}_protein.pdb"); st.setup_entities(); st.remove_ligands_and_waters()
    chains, patoms = [], []
    L = ref.GetConformer().GetPositions()
    pocket_res = []
    for ch in st[0]:
        ca = []
        for r in ch:
            t = gemmi.find_tabulated_residue(r.name)
            if not (t and t.is_amino_acid()): continue
            a = r.find_atom("CA", "*")
            if a: ca.append(np.array(a.pos.tolist()) - center)
            for at in r:
                p = np.array(at.pos.tolist())
                if np.linalg.norm(p - L, axis=1).min() < 9.0:
                    patoms.append((p - center, r.name))
            dmin = min(np.linalg.norm(np.array(at.pos.tolist()) - L, axis=1).min() for at in r)
            if dmin < 4.0 and a is not None:
                pocket_res.append(dict(n=f"{r.name}{r.seqid.num}", p=np.round(np.array(a.pos.tolist()) - center, 2).tolist(), d=round(float(dmin), 2)))
        if len(ca) >= 10:
            ca = np.array(ca); chains.append(segments(ca, ss_assign(ca)))
    # pocket surface: gaussian density of pocket atoms, marching cubes, coloured by residue class
    P = np.array([p for p, _ in patoms]); cls = [n for _, n in patoms]
    lo, hi = P.min(0) - 3, P.max(0) + 3; h = 0.7
    g = np.mgrid[lo[0]:hi[0]:h, lo[1]:hi[1]:h, lo[2]:hi[2]:h]; grid = np.stack(g, -1)
    dens = np.zeros(grid.shape[:3])
    for p in P: dens += np.exp(-((grid - p) ** 2).sum(-1) / (2 * 1.1 ** 2))
    # carve out the ligand site so the surface wraps the cavity, not the ligand
    for p in L - center: dens -= 0.0 * np.exp(-((grid - p) ** 2).sum(-1))
    v, f, _, _ = measure.marching_cubes(dens, level=0.55, step_size=2)
    v = v * h + lo
    near = np.argmin(((v[:, None, :] - P[None, :, :]) ** 2).sum(-1), axis=1)
    col = [0 if cls[k] in HYD else 1 if cls[k] in POS else 2 if cls[k] in NEG else 3 for k in near]
    # keep only the cavity-facing patch within 12 A of the ligand centroid
    keep_v = np.linalg.norm(v, axis=1) < 12.0
    fk = f[keep_v[f].all(1)]
    used = np.unique(fk); remap = -np.ones(len(v), int); remap[used] = np.arange(len(used))
    out = dict(id=i, rmsd=row.get("rmsd"), success=row.get("success"), pb_valid=row.get("pb_valid"), pb_failed=row.get("pb_failed", []),
               confidence=row.get("confidence"), ligand_iptm=row.get("ligand_iptm"), pocket_ca_rmsd=row.get("pocket_ca_rmsd"),
               chains=chains, xtal=mol_json(ref, center), pocket=pocket_res[:40],
               surf=dict(v=np.round(v[used], 2).ravel().tolist(), f=remap[fk].ravel().tolist(), c=[col[k] for k in used]),
               smiles=Chem.MolToSmiles(ref))
    pf = f"{EV}/{i}/pred_ligand.sdf"
    if os.path.exists(pf):
        pred = Chem.MolFromMolFile(pf)
        match = pred.GetSubstructMatch(ref)  # pred atom index for each crystal atom
        if match:
            pp = pred.GetConformer().GetPositions()[list(match)] - center
            out["pred"] = np.round(pp, 3).tolist()
    return out

poses = json.load(open(f"{EV}/poses.json"))
rows = {r["id"]: r for r in poses["per_complex"]}
ok = [r for r in poses["per_complex"] if r.get("status") == "ok"]
# showcase: best confident success, a second success, and the worst miss (honest failure)
succ = sorted([r for r in ok if r.get("success")], key=lambda r: -(r.get("confidence") or 0))
miss = sorted([r for r in ok if not r.get("success")], key=lambda r: -(r.get("rmsd") or 0))
show = [r["id"] for r in succ[:2]] + [r["id"] for r in miss[:1]]
complexes = [complex_data(i, rows[i]) for i in show]

aff = json.load(open(f"{RES}/affinity.json"))
ligs = []
for t, d in aff["targets"].items():
    for l in d["ligands"]: ligs.append(dict(t=t, **l))
smiles = {}
from rdkit.Chem import SDMolSupplier
for t in aff["targets"]:
    for m in SDMolSupplier(f"{RES}/../fep_inputs/{t}_ligands.sdf"):
        if m: smiles[(t, m.GetProp("_Name"))] = Chem.MolToSmiles(m)
fps = []
for l in ligs:
    m = Chem.MolFromSmiles(smiles[(l["t"], l["name"])]); l["smiles"] = smiles[(l["t"], l["name"])]
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048); a = np.zeros(2048); DataStructs.ConvertToNumpyArray(fp, a); fps.append(a)
from sklearn.decomposition import PCA
fps = np.array(fps)
for k, t in enumerate(sorted({l["t"] for l in ligs}, reverse=True)):
    idx = [j for j, l in enumerate(ligs) if l["t"] == t]
    xyz = PCA(3, random_state=0).fit_transform(fps[idx]); xyz = xyz / np.abs(xyz).max(0) * 5.5
    for j, p in zip(idx, xyz): ligs[j]["xyz"] = np.round(p + np.array([-9 + 18 * k, 0, 0]), 3).tolist()

data = dict(stamp=json.load(open(f"{RES}/stamp.json")), bench=poses["summary"], published=json.load(open(f"{RES}/published.json")),
            complexes=complexes, affinity=dict(ligands=ligs, metrics={t: {k: v for k, v in d.items() if k != "ligands"} for t, d in aff["targets"].items()}))
tpl = open(f"{HERE}/viewer_template.html").read()
html = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
open(f"{HERE}/viewer.html", "w").write(html)
print("viewer.html", round(len(html) / 1e6, 2), "MB; showcase", show)
