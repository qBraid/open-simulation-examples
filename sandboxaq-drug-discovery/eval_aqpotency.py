"""Score SandboxAQ's hosted AQPotency on the same TYK2/CDK2 ligands as Boltz-2 and FEP+.

AQPotency returns pIC50 (higher = more potent), converted with
dG = -1.364 * pIC50 kcal/mol (298 K, IC50 ~ Kd) so it sits on the same axis.
Predictions are matched to ligand names by RDKit canonical SMILES. Also reports
how many test ligands AQPotency flags as seen in training (ligand_similarity = 1).

usage: eval_aqpotency.py <aqpotency_raw.json> <inputs.csv> <out.json>
"""
import csv, json, sys
import numpy as np
from rdkit import Chem
from scipy import stats

RAW, INP, OUT = sys.argv[1:4]
raw = json.load(open(RAW))
canon = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))

def metrics(x, y, seed=0):  # identical to drug-discovery/eval_affinity.py
    rng = np.random.default_rng(seed); x, y = np.asarray(x), np.asarray(y)
    out = dict(pearson=stats.pearsonr(x, y)[0], spearman=stats.spearmanr(x, y)[0], kendall=stats.kendalltau(x, y)[0],
               rmse_centered=float(np.sqrt(np.mean(((x - x.mean()) - (y - y.mean())) ** 2))))
    b = []
    for _ in range(2000):
        k = rng.integers(0, len(x), len(x))
        if len(set(k)) > 3: b.append([stats.spearmanr(x[k], y[k])[0], stats.kendalltau(x[k], y[k])[0]])
    b = np.array(b)
    out["spearman_ci95"] = [float(np.nanpercentile(b[:, 0], 2.5)), float(np.nanpercentile(b[:, 0], 97.5))]
    out["kendall_ci95"] = [float(np.nanpercentile(b[:, 1], 2.5)), float(np.nanpercentile(b[:, 1], 97.5))]
    return {k: (float(v) if not isinstance(v, list) else v) for k, v in out.items()}

rows = list(csv.DictReader(open(INP)))
res = dict(method=__doc__, source=raw["source"], jobs=raw["jobs"], targets={})
for t in ("tyk2", "cdk2"):
    pred = {canon(s): dict(pic50=m, sigma=sg, sim=sim) for s, m, sg, sim in raw[t]}
    lig = []
    for r in (r for r in rows if r["target"] == t):
        p = pred[canon(r["smiles"])]
        lig.append(dict(name=r["ligand"], exp=float(r["exp_dG_kcal"]), aqpotency=-1.364 * p["pic50"], **p))
    e = [l["exp"] for l in lig]
    res["targets"][t] = dict(n=len(lig), seen_in_training=sum(l["sim"] >= 0.999 for l in lig),
                             median_sigma_pic50=float(np.median([l["sigma"] for l in lig])),
                             aqpotency=metrics([l["aqpotency"] for l in lig], e), ligands=lig)
    m = res["targets"][t]
    print(t, "n", m["n"], "seen", m["seen_in_training"], "spearman %.2f kendall %.2f rmse %.2f" % (
        m["aqpotency"]["spearman"], m["aqpotency"]["kendall"], m["aqpotency"]["rmse_centered"]))
json.dump(res, open(OUT, "w"), indent=1)
