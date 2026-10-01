"""Run an Earth2Studio forecast model from ERA5 initial conditions and score it
against ERA5 the WeatherBench2 way (latitude-weighted global RMSE, 0.25 deg).

Usage: python forecast_score.py <model: sfno|fcn3> <out_dir> [--viz-init 2020-08-24T00]
Writes <out_dir>/<model>_scores.json and, for the viz init, a compact npz of
fields at every 6 h step for the viewer.
"""
import argparse, json, os, time
from datetime import datetime, timedelta

import numpy as np
import torch

from earth2studio.data import ARCO, fetch_data
from earth2studio.models.px import SFNO

INITS = ["2020-01-15", "2020-02-05", "2020-03-15", "2020-04-15", "2020-05-17", "2020-06-15",
         "2020-07-15", "2020-08-24", "2020-09-14", "2020-10-15", "2020-11-14", "2020-12-15"]
SCORE_VARS = ["z500", "t850", "t2m"]
LEADS_H = [24, 72, 120]
VIZ_VARS = ["z500", "t2m", "u850", "v850", "msl", "tcwv"]


def load_model(name):
    if name == "sfno":
        return SFNO.load_model(SFNO.load_default_package())
    if name == "fcn3":
        from earth2studio.models.px import FCN3
        return FCN3.load_model(FCN3.load_default_package())
    raise ValueError(name)


def lat_weights(lat):
    w = np.cos(np.deg2rad(lat))
    return w / w.mean()


def rmse(a, b, w):
    return float(np.sqrt(((a - b) ** 2 * w[:, None]).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--viz-init", default="2020-08-24T00")
    ap.add_argument("--inits", default=",".join(INITS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    t0 = time.time()
    model = load_model(a.model).to(dev)
    load_s = time.time() - t0
    era5 = ARCO()
    nsteps = max(LEADS_H) // 6
    out = {"model": a.model, "inits": [], "rmse": {v: {str(h): [] for h in LEADS_H} for v in SCORE_VARS},
           "load_s": round(load_s, 1), "step_s": []}
    viz = None
    for d in a.inits.split(","):
        init = datetime.fromisoformat(d) if "T" in d else datetime.fromisoformat(d + "T00:00")
        x, coords = fetch_data(era5, time=np.array([np.datetime64(init)]), variable=model.input_coords()["variable"],
                               lead_time=model.input_coords()["lead_time"], device=dev)
        varlist = list(coords["variable"])
        is_viz = init == datetime.fromisoformat(a.viz_init)
        keep = {}
        it = model.create_iterator(x, coords)
        ts = time.time()
        for step, (xs, cs) in enumerate(it):
            h = step * 6
            vars_now = list(cs["variable"])
            if h in LEADS_H or is_viz:
                for v in set(SCORE_VARS + (VIZ_VARS if is_viz else [])):
                    if v in vars_now:
                        keep[(v, h)] = (xs[0, 0, 0, vars_now.index(v)] if xs.dim() == 6 else xs[0, 0, vars_now.index(v)]).float().cpu().numpy()
            if step >= nsteps:
                break
        torch.cuda.synchronize(); out["step_s"].append((time.time() - ts) / nsteps)
        lat = np.asarray(cs["lat"]); w = lat_weights(lat)
        for v in SCORE_VARS:
            for h in LEADS_H:
                vt = np.datetime64(init + timedelta(hours=h))
                truth, _ = fetch_data(era5, time=np.array([vt]), variable=np.array([v]), device="cpu")
                tr = truth[0, 0, 0].numpy()
                out["rmse"][v][str(h)].append(rmse(keep[(v, h)], tr, w))
        out["inits"].append(init.isoformat())
        print(d, {v: [round(out["rmse"][v][str(h)][-1], 2) for h in LEADS_H] for v in SCORE_VARS}, flush=True)
        if is_viz:
            viz = {f"{v}_{h:03d}": keep[(v, h)] for (v, h) in keep}
            # matching ERA5 truth every 12 h for the swipe and error map
            for h in range(0, max(LEADS_H) + 1, 12):
                vt = np.datetime64(init + timedelta(hours=h))
                tr, _ = fetch_data(era5, time=np.array([vt]), variable=np.array(["z500", "t2m", "u850", "v850"]), device="cpu")
                for k, v in enumerate(["z500", "t2m", "u850", "v850"]):
                    viz[f"era5_{v}_{h:03d}"] = tr[0, 0, k].numpy()
            np.savez_compressed(os.path.join(a.out, f"{a.model}_viz_{init:%Y%m%d%H}.npz"), lat=lat, lon=np.asarray(cs["lon"]),
                                **{k: v.astype(np.float32) for k, v in viz.items()})
        json.dump(out, open(os.path.join(a.out, f"{a.model}_scores.json"), "w"), indent=1)
    out["mean"] = {v: {h: float(np.mean(out["rmse"][v][h])) for h in out["rmse"][v]} for v in SCORE_VARS}
    out["gpu"] = torch.cuda.get_device_name(0); out["wall_s"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(a.out, f"{a.model}_scores.json"), "w"), indent=1)
    print("MEAN", out["mean"], "wall", out["wall_s"])


if __name__ == "__main__":
    main()
