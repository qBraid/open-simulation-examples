"""Score discovery predictions against WBM DFT and against the published predictions for the same model.

usage: score_discovery.py <model: mp0|mpa0> <out.json> <pred1.csv> [pred2.csv ...]
"""
import sys, json
import numpy as np, pandas as pd

model, out, files = sys.argv[1], sys.argv[2], sys.argv[3:]
s = pd.read_csv("data/wbm-summary.csv.gz").set_index("material_id")
pub = pd.read_csv(f"data/{ {'mp0': 'mace-mp-0', 'mpa0': 'mace-mpa-0'}[model] }-discovery.csv.gz").set_index("material_id").e_form_per_atom
p = pd.concat([pd.read_csv(f) for f in files]).dropna(subset=["energy"]).set_index("material_id")
d = s.loc[p.index]
e_form_true = d.e_form_per_atom_mp2020_corrected
e_hull_true = d.e_above_hull_mp2020_corrected_ppd_mp
e_form_pred = e_form_true + (p.energy - d.uncorrected_energy_from_cse) / d.n_sites
e_hull_pred = e_hull_true + (e_form_pred - e_form_true)
e_hull_pub = e_hull_true + (pub.loc[p.index] - e_form_true)


def metrics(pred, true):
    tp = ((pred <= 0) & (true <= 0)).sum(); fp = ((pred <= 0) & (true > 0)).sum()
    fn = ((pred > 0) & (true <= 0)).sum(); tn = ((pred > 0) & (true > 0)).sum()
    prec = tp / (tp + fp); rec = tp / (tp + fn)
    err = pred - true
    return dict(F1=float(2 * prec * rec / (prec + rec)), Precision=float(prec), Recall=float(rec),
                Accuracy=float((tp + tn) / len(true)), DAF=float(prec / (true <= 0).mean()),
                MAE=float(err.abs().mean()), RMSE=float(np.sqrt((err ** 2).mean())),
                R2=float(1 - (err ** 2).sum() / ((true - true.mean()) ** 2).sum()),
                TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn))


rng = np.random.default_rng(0)
boot = []
for _ in range(1000):
    idx = rng.integers(0, len(p), len(p))
    boot.append(metrics(e_hull_pred.iloc[idx], e_hull_true.iloc[idx])["F1"])
res = dict(model=model, n=int(len(p)), n_failed=int(sum(pd.concat([pd.read_csv(f) for f in files]).energy.isna())),
           ours=metrics(e_hull_pred, e_hull_true), published_same_structures=metrics(e_hull_pub, e_hull_true),
           F1_bootstrap_95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
           per_structure_vs_published=dict(MAE=float((e_form_pred - pub.loc[p.index]).abs().mean()),
                                           median=float((e_form_pred - pub.loc[p.index]).abs().median()),
                                           frac_within_10meV=float(((e_form_pred - pub.loc[p.index]).abs() < 0.01).mean())),
           mean_relax_sec=float(p.sec.mean()), mean_steps=float(p.steps.mean()))
json.dump(res, open(out, "w"), indent=1)
pd.DataFrame(dict(e_hull_true=e_hull_true, e_hull_pred=e_hull_pred, formula=d.formula)).to_csv(out.replace(".json", "_parity.csv"))
print(json.dumps(res, indent=1))
