"""AQAffinity vs Boltz-2 vs FEP+ vs experiment on the identical TYK2/CDK2 ligands.

AQAffinity writes one affinity per structure sample (<query>_seed_<s>_sample_<k>_binding_head.txt).
Primary score: the sample OpenFold3 ranks highest (sample_ranking_score), the same
"top structure" rule Boltz-2 uses. Secondary: the mean over all samples.
Sign convention: higher output = stronger binding, i.e. -log10(IC50 / uM) = pIC50 - 6.
This is the opposite of Boltz-2's affinity_pred_value. Evidence: the decoy loss
(modules/model/losses.py) penalises decoys unless the prediction is below -3, and
the regression labels have median +0.96 (pIC50 ~7, typical of ChEMBL actives).
So dG = -1.364 * (value + 6) kcal/mol. Ranking metrics do not depend on the offset.
Metrics, bootstrap and centring are identical to drug-discovery/eval_affinity.py.

usage: eval_aqaffinity.py <run_dir>[:label] ... <fep_dir> <boltz_affinity.json> <out.json>
(fep_dir holds fep_plus_tyk2.csv and fep_plus_cdk2.csv, as in drug-discovery/results)
"""
import csv, glob, json, os, re, sys
import numpy as np
from scipy import stats

*RUNS, FEP, BOLTZ, OUT = sys.argv[1:]
boltz = json.load(open(BOLTZ))["targets"]

def metrics(x, y, seed=0):
    rng = np.random.default_rng(seed)
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

def read_run(d):
    """query -> dict(top=value of top-ranked sample, mean=mean over samples, n=samples, ranking=score)"""
    res = {}
    for qdir in glob.glob(f"{d}/*__*/"):
        q = os.path.basename(qdir.rstrip("/"))
        vals, rank = {}, {}
        for f in glob.glob(f"{qdir}/**/*_binding_head.txt", recursive=True):
            k = int(re.search(r"_sample_(\d+)_binding_head", f).group(1)); vals[k] = float(open(f).read())
        for f in glob.glob(f"{qdir}/**/*_confidences_aggregated.json", recursive=True):
            m = re.search(r"_sample_(\d+)_confidences_aggregated", f)
            if m: rank[int(m.group(1))] = json.load(open(f)).get("sample_ranking_score")
        if not vals: continue
        ranked = [k for k in vals if rank.get(k) is not None]
        top = max(ranked, key=lambda k: rank[k]) if ranked else min(vals)
        res[q] = dict(top=vals[top], top_sample=top, top_ranking=rank.get(top), mean=float(np.mean(list(vals.values()))),
                      spread=float(np.std(list(vals.values()))), n=len(vals))
    return res

runs = []
for r in RUNS:
    d, _, label = r.partition(":"); runs.append((label or os.path.basename(d), read_run(d)))

out = dict(method=__doc__, runs=[l for l, _ in runs], primary_run=runs[0][0], targets={})  # first run listed is the headline
for t in ("tyk2", "cdk2"):
    rows = list(csv.DictReader(open(f"{FEP}/fep_plus_{t}.csv")))
    bz = {l["name"]: l["boltz"] for l in boltz[t]["ligands"]}
    lig = []
    for r in rows:
        n = r["Ligand name"]; e = dict(name=n, exp=float(r["Exp. dG (kcal/mol)"]), fep=float(r["Pred. dG (kcal/mol)"]), boltz=bz.get(n))
        for label, res in runs:
            v = res.get(f"{t}__{n}")
            if v: e[label] = -1.364 * (v["top"] + 6); e[label + "_mean"] = -1.364 * (v["mean"] + 6); e[label + "_raw"] = v
        lig.append(e)
    exp = [l["exp"] for l in lig]
    m = dict(n=len(lig), exp_range_kcal=float(max(exp) - min(exp)), fep_plus=metrics([l["fep"] for l in lig], exp),
             boltz2=metrics([l["boltz"] for l in lig], exp))
    for label, _ in runs:
        have = [l for l in lig if label in l]
        if len(have) >= 4:
            ee = [l["exp"] for l in have]
            m[label] = dict(n=len(have), top=metrics([l[label] for l in have], ee), mean=metrics([l[label + "_mean"] for l in have], ee),
                            vs_boltz2_spearman=float(stats.spearmanr([l[label] for l in have], [l["boltz"] for l in have])[0]))
    out["targets"][t] = dict(metrics=m, ligands=lig)
    line = f"{t} n={len(lig)} FEP+ {m['fep_plus']['spearman']:.2f} Boltz-2 {m['boltz2']['spearman']:.2f}"
    for label, _ in runs:
        if label in m: line += f" | {label} top {m[label]['top']['spearman']:.2f} mean {m[label]['mean']['spearman']:.2f} (n={m[label]['n']})"
    print(line)
json.dump(out, open(OUT, "w"), indent=1)
