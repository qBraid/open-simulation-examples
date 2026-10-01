"""Score ECMWF IFS HRES (WeatherBench2 public copy) on the same init dates and the
same ERA5 truth as forecast_score.py, so the comparison is paired.

Usage: python hres_score.py <out_dir> [--viz-init 2020-08-24T00]
"""
import argparse, json, os
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from earth2studio.data import ARCO, fetch_data
from forecast_score import INITS, LEADS_H, SCORE_VARS, lat_weights, rmse

HRES = "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
NAMES = {"z500": ("geopotential", 500), "t850": ("temperature", 850), "t2m": ("2m_temperature", None)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--viz-init", default="2020-08-24T00")
    a = ap.parse_args()
    ds = xr.open_zarr(HRES, storage_options={"token": "anon"})
    era5 = ARCO(cache=False)
    out = {"model": "ifs_hres", "inits": [], "rmse": {v: {str(h): [] for h in LEADS_H} for v in SCORE_VARS}}
    viz = {}
    for d in INITS:
        init = datetime.fromisoformat(d + "T00:00")
        for v in SCORE_VARS:
            name, lev = NAMES[v]
            for h in LEADS_H:
                # newer xarray leaves WB2's prediction_timedelta as integer hours
                lead = h if ds.prediction_timedelta.dtype.kind in "iu" else np.timedelta64(h, "h")
                da = ds[name].sel(time=np.datetime64(init), prediction_timedelta=lead)
                if lev: da = da.sel(level=lev)
                f = da.transpose("latitude", "longitude").values
                if ds.latitude.values[0] < ds.latitude.values[-1]:  # WB2 is south->north; ERA5 ARCO is north->south
                    f = f[::-1]
                tr, c = fetch_data(era5, time=np.array([np.datetime64(init + timedelta(hours=h))]), variable=np.array([v]), device="cpu")
                lat = np.asarray(c["lat"])
                out["rmse"][v][str(h)].append(rmse(f, tr[0, 0, 0].numpy(), lat_weights(lat)))
                if init == datetime.fromisoformat(a.viz_init) and v == "z500":
                    viz[f"hres_z500_{h:03d}"] = f.astype(np.float32)
        out["inits"].append(init.isoformat())
        print(d, {v: [round(out["rmse"][v][str(h)][-1], 2) for h in LEADS_H] for v in SCORE_VARS}, flush=True)
    out["mean"] = {v: {h: float(np.mean(out["rmse"][v][h])) for h in out["rmse"][v]} for v in SCORE_VARS}
    json.dump(out, open(os.path.join(a.out, "hres_scores.json"), "w"), indent=1)
    if viz: np.savez_compressed(os.path.join(a.out, "hres_viz.npz"), **viz)
    print("MEAN", out["mean"])


if __name__ == "__main__":
    main()
