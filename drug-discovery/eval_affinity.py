"""Boltz-2 affinity vs experiment vs FEP+ on identical ligands (TYK2, CDK2).

Boltz-2 reports affinity_pred_value = log10(IC50 / uM); converted to a free
energy with dG = RT ln(10) * (value - 6) = 1.364 * (value - 6) kcal/mol at
298 K (treating IC50 ~ Kd, the same approximation the Boltz-2 paper uses).
Ranking metrics do not depend on that conversion. RMSE is reported after
removing the mean offset (a relative-binding comparison, as for FEP+).
95% intervals: 2000 bootstrap resamples of the ligands.
"""
import csv, glob, json, sys
import numpy as np
from scipy import stats

PRED, FEP, OUT = sys.argv[1:4]
rng = np.random.default_rng(0)

def metrics(x, y):
    x, y = np.asarray(x), np.asarray(y)
    out = dict(pearson=stats.pearsonr(x, y)[0], spearman=stats.spearmanr(x, y)[0], kendall=stats.kendalltau(x, y)[0],
               rmse_centered=float(np.sqrt(np.mean(((x - x.mean()) - (y - y.mean())) ** 2))))
    boot = []
    for _ in range(2000):
        k = rng.integers(0, len(x), len(x))
        if len(set(k)) > 3: boot.append([stats.spearmanr(x[k], y[k])[0], stats.kendalltau(x[k], y[k])[0]])
    b = np.array(boot)
    out["spearman_ci95"] = [float(np.nanpercentile(b[:, 0], 2.5)), float(np.nanpercentile(b[:, 0], 97.5))]
    out["kendall_ci95"] = [float(np.nanpercentile(b[:, 1], 2.5)), float(np.nanpercentile(b[:, 1], 97.5))]
    return {k: (float(v) if not isinstance(v, list) else v) for k, v in out.items()}

res = {}
for t in ("tyk2", "cdk2"):
    rows = list(csv.DictReader(open(f"{FEP}/{t}_out.csv")))
    lig = []
    for r in rows:
        f = glob.glob(f"{PRED}/**/affinity_{t}__{r['Ligand name']}.json", recursive=True)
        if not f: continue
        a = json.load(open(f[0]))
        lig.append(dict(name=r["Ligand name"], exp=float(r["Exp. dG (kcal/mol)"]), fep=float(r["Pred. dG (kcal/mol)"]),
                        boltz=1.364 * (a["affinity_pred_value"] - 6), p_binder=a.get("affinity_probability_binary")))
    if len(lig) < 4: continue
    e = [l["exp"] for l in lig]
    res[t] = dict(n=len(lig), boltz2=metrics([l["boltz"] for l in lig], e), fep_plus=metrics([l["fep"] for l in lig], e),
                  exp_range_kcal=float(max(e) - min(e)), ligands=lig)
    print(t, len(lig), "Boltz-2 spearman %.2f kendall %.2f | FEP+ spearman %.2f kendall %.2f" % (
        res[t]["boltz2"]["spearman"], res[t]["boltz2"]["kendall"], res[t]["fep_plus"]["spearman"], res[t]["fep_plus"]["kendall"]))
json.dump(dict(method=__doc__, targets=res), open(OUT, "w"), indent=1)
