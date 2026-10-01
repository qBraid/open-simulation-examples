"""FCN3 is a probabilistic model: score an N-member ensemble mean (and the members) against ERA5,
paired with IFS HRES on the same starts.

Usage: python ensemble_score.py <out_dir> --members 4 --inits 2020-01-15,2020-03-15,...
Writes <out_dir>/fcn3_ens_scores.json.
"""
import argparse, json, os, time
from datetime import datetime, timedelta

import numpy as np
import torch

from earth2studio.data import ARCO, fetch_data
from earth2studio.models.px import FCN3
from forecast_score import LEADS_H, SCORE_VARS, lat_weights, rmse

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--members", type=int, default=4)
ap.add_argument("--inits", default="2020-01-15,2020-03-15,2020-05-17,2020-07-15,2020-09-14,2020-11-14")
a = ap.parse_args()
dev = torch.device("cuda"); t0 = time.time()
model = FCN3.load_model(FCN3.load_default_package()).to(dev)
era5 = ARCO(cache=False); nsteps = max(LEADS_H) // 6
out = {"model": "fcn3_ens", "members": a.members, "inits": [],
       "rmse_mean": {v: {str(h): [] for h in LEADS_H} for v in SCORE_VARS},
       "rmse_member": {v: {str(h): [] for h in LEADS_H} for v in SCORE_VARS}, "step_s": []}
for d in a.inits.split(","):
    init = datetime.fromisoformat(d + "T00:00")
    x, coords = fetch_data(era5, time=np.array([np.datetime64(init)]), variable=model.input_coords()["variable"],
                           lead_time=model.input_coords()["lead_time"], device=dev)
    truth, lat = {}, None
    for v in SCORE_VARS:
        for h in LEADS_H:
            tr, c = fetch_data(era5, time=np.array([np.datetime64(init + timedelta(hours=h))]), variable=np.array([v]), device="cpu")
            truth[(v, h)] = tr[0, 0, 0].numpy(); lat = np.asarray(c["lat"])
    w = lat_weights(lat)
    acc = {}
    for m in range(a.members):
        model.set_rng(seed=1000 + m, reset=True)
        ts = time.time()
        for step, (xs, cs) in enumerate(model.create_iterator(x, coords)):
            h = step * 6
            if h in LEADS_H:
                vn = list(cs["variable"])
                for v in SCORE_VARS:
                    f = (xs[0, 0, 0, vn.index(v)] if xs.dim() == 6 else xs[0, 0, vn.index(v)]).float().cpu().numpy()
                    acc[(v, h)] = acc.get((v, h), 0) + f
                    out["rmse_member"][v][str(h)].append(rmse(f, truth[(v, h)], w))
            if step >= nsteps: break
        torch.cuda.synchronize(); out["step_s"].append((time.time() - ts) / nsteps)
    for v in SCORE_VARS:
        for h in LEADS_H:
            out["rmse_mean"][v][str(h)].append(rmse(acc[(v, h)] / a.members, truth[(v, h)], w))
    out["inits"].append(init.isoformat())
    print(d, {v: [round(out["rmse_mean"][v][str(h)][-1], 2) for h in LEADS_H] for v in SCORE_VARS}, flush=True)
    json.dump(out, open(os.path.join(a.out, "fcn3_ens_scores.json"), "w"), indent=1)
out["mean"] = {v: {h: float(np.mean(out["rmse_mean"][v][h])) for h in out["rmse_mean"][v]} for v in SCORE_VARS}
out["wall_s"] = round(time.time() - t0, 1); out["gpu"] = torch.cuda.get_device_name(0)
json.dump(out, open(os.path.join(a.out, "fcn3_ens_scores.json"), "w"), indent=1)
print("MEAN", out["mean"], "wall", out["wall_s"])
