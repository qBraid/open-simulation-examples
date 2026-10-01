"""Build the self-contained globe viewer from forecast/ERA5 fields and scores.

Usage: python build_viewer.py <results_dir> <viz.npz> [--out viewer.html]
Fields are downsampled to 1 deg, quantized to 8 bit (winds int8 at 0.5 m/s) and
deflate-compressed; the page inflates them with the browser's DecompressionStream.
"""
import argparse, base64, json, os, zlib
from datetime import datetime, timedelta

import numpy as np
import cmocean
import contourpy

from land_rings import pack

HERE = os.path.dirname(os.path.abspath(__file__))
VMETA = {
    "tcwv": dict(label="Total column water vapour", units="kg/m²", lo=0, hi=70, cmap="ice", err=12),
    "t2m": dict(label="2 m temperature", units="°C", lo=-50, hi=45, cmap="thermal", err=5),
    "msl": dict(label="Sea-level pressure", units="hPa", lo=960, hi=1045, cmap="deep_r", err=8),
}
CONV = {"tcwv": lambda a: a, "t2m": lambda a: a - 273.15, "msl": lambda a: a / 100.0}
WIND_SCALE = 0.5  # m/s per int8 step
STORMS = [  # name, approx position at init (lon, lat), search radius deg
    ("Hurricane Laura", -72.0, 18.0, 6.0),
    ("Typhoon Bavi", 125.5, 24.0, 6.0),
]


def b64z(arr):
    return base64.b64encode(zlib.compress(np.ascontiguousarray(arr).tobytes(), 9)).decode()


def down(a):  # 0.25 deg (721 x 1440, lon 0..359.75) -> 1 deg (181 x 360)
    return a[::4, ::4]


def q8(a, lo, hi):
    return np.clip(np.round((a - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)


def lut(name):
    cm = cmocean.cm.cmap_d[name]
    rgb = (np.array([cm(i / 255)[:3] for i in range(256)]) * 255).round().astype(np.uint8)
    return base64.b64encode(rgb.tobytes()).decode()


def contours(z, levels):
    """z500 height (m) on the 1 deg grid -> list of (n_points, level) and int16 lon/lat*100 points."""
    lat = np.linspace(90, -90, z.shape[0]); lon = np.arange(z.shape[1] + 1) * 1.0
    zz = np.concatenate([z, z[:, :1]], axis=1)  # wrap
    gen = contourpy.contour_generator(lon, lat, zz)
    meta, pts = [], []
    for lev in levels:
        for line in gen.lines(lev):
            if len(line) < 4: continue
            line = line[:: max(1, len(line) // 400)]  # cap points per line
            lo = np.where(line[:, 0] > 180, line[:, 0] - 360, line[:, 0])
            pts.append(np.stack([lo, line[:, 1]], 1)); meta.append([len(line), int(lev)])
    p = np.round(np.concatenate(pts) * 100).astype(np.int16) if pts else np.zeros((0, 2), np.int16)
    return meta, p


def track(frames, lon0, lat0, rad, lat, lon):
    """Follow a sea-level-pressure minimum through frames (0.25 deg, Pa)."""
    out = []; clon, clat = lon0 % 360, lat0
    for f in frames:
        la = np.abs(lat - clat) <= rad
        dl = np.abs(((lon - clon + 180) % 360) - 180) <= rad
        sub = np.where(la[:, None] & dl[None, :], f, np.inf)
        i, j = np.unravel_index(np.argmin(sub), sub.shape)
        p = f[i, j] / 100
        if p > 1006: break
        clat, clon = float(lat[i]), float(lon[j])
        out.append([round(((clon + 180) % 360) - 180, 2), round(clat, 2), round(float(p), 1)])
        rad = 4.0
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("results"); ap.add_argument("viz"); ap.add_argument("--out", default=os.path.join(HERE, "viewer.html")); ap.add_argument("--init", default="2020-08-24T00:00")
    a = ap.parse_args()
    Z = np.load(a.viz)
    lat, lon = Z["lat"], Z["lon"]
    leads = list(range(0, 121, 6)); tleads = list(range(0, 121, 12))
    blobs, vmeta = {}, {}
    for v, m in VMETA.items():
        fc = np.stack([q8(CONV[v](down(Z[f"{v}_{h:03d}"])), m["lo"], m["hi"]) for h in leads])
        tr = np.stack([q8(CONV[v](down(Z[f"era5_{v}_{h:03d}"])), m["lo"], m["hi"]) for h in tleads])
        blobs[f"fc_{v}"] = b64z(fc); blobs[f"tr_{v}"] = b64z(tr); vmeta[v] = m
    for c in ("u850", "v850"):
        blobs[f"fc_{c}"] = b64z(np.stack([np.clip(np.round(down(Z[f"{c}_{h:03d}"]) / WIND_SCALE), -127, 127).astype(np.int8) for h in leads]))
        blobs[f"tr_{c}"] = b64z(np.stack([np.clip(np.round(down(Z[f"era5_{c}_{h:03d}"]) / WIND_SCALE), -127, 127).astype(np.int8) for h in tleads]))
    levels = list(range(4800, 6001, 60))
    z5 = {}
    for kind, keyf, ls in (("fc", "z500_{:03d}", leads), ("tr", "era5_z500_{:03d}", tleads)):
        metas, allp = [], []
        for h in ls:
            m_, p_ = contours(down(Z[keyf.format(h)]) / 9.80665, levels); metas.append(m_); allp.append(p_)
        z5[kind] = metas; blobs[f"z5_{kind}_pts"] = b64z(np.concatenate(allp))
    land_pts, land_lens = pack(os.path.join(HERE, "data", "land-50m.json"))
    blobs["land_pts"] = b64z(land_pts)

    S = json.load(open(os.path.join(a.results, "sfno_scores.json")))
    H = json.load(open(os.path.join(a.results, "hres_scores.json")))
    R = json.load(open(os.path.join(a.results, "wb2_ref_2020.json")))
    init = datetime.fromisoformat(a.init)
    # storm tracks: forecast every 6 h, ERA5 every 12 h
    tracks = []
    for name, lo0, la0, rad in STORMS:
        fc = track([Z[f"msl_{h:03d}"] for h in leads], lo0, la0, rad, lat, lon)
        tr = track([Z[f"era5_msl_{h:03d}"] for h in tleads], lo0, la0, rad, lat, lon)
        tracks.append({"name": name, "fc": fc, "tr": tr})

    def series(var):
        k = ["24", "72", "120"]
        sf = S["rmse"][var]; hp = H["rmse"][var]
        return [
            {"label": "SFNO (this run, 12 inits)", "color": "--c-model", "bold": True,
             "vals": [float(np.mean(sf[h])) for h in k], "band": [[float(np.min(sf[h])), float(np.max(sf[h]))] for h in k]},
            {"label": "IFS HRES, same 12 inits", "color": "--c-hres", "vals": [float(np.mean(hp[h])) for h in k]},
            {"label": "IFS HRES, WB2 2020 (730 inits)", "color": "--c-hres", "dash": True, "vals": [R["hres"][var].get(h) for h in k]},
            {"label": "GraphCast, WB2 2020", "color": "--c-gc", "dash": True, "vals": [R["graphcast"][var].get(h) for h in k]},
            {"label": "Pangu, WB2 2020", "color": "--c-pangu", "dash": True, "vals": [R["pangu"][var].get(h) for h in k]},
        ]

    rel = {v: (np.mean(S["rmse"][v]["120"]) / np.mean(H["rmse"][v]["120"]) - 1) * 100 for v in ("z500", "t850", "t2m")}
    verdict = json.load(open(os.path.join(a.results, "verdict.json"))) if os.path.exists(os.path.join(a.results, "verdict.json")) else {}
    payload = {
        "model_label": "NVIDIA SFNO (FourCastNet v2, 73 ch)", "init": init.isoformat()[:16], "gpu": S.get("gpu", "NVIDIA L4"),
        "step_s": float(np.mean(S["step_s"])), "grid": {"nx": 360, "ny": 181}, "leads": leads, "truth_leads": tleads,
        "vars": list(VMETA), "vmeta": vmeta, "wind_scale": WIND_SCALE,
        "cmaps": {"ice": lut("ice"), "thermal": lut("thermal"), "deep_r": lut("deep_r"), "diverging": lut("balance")},
        "z5_fc": z5["fc"], "z5_tr": z5["tr"], "land_lens": land_lens.tolist(), "tracks": tracks,
        "scores": {v: series(v) for v in ("z500", "t850", "t2m")},
        "score_labels": {"z500": "z500 (m²/s²)", "t850": "T850 (K)", "t2m": "T2m (K)"},
        "verdict_html": verdict.get("html", ""), "stamp_html": verdict.get("stamp_html", ""),
        "tour": verdict.get("tour", []), "rel_day5_vs_hres_pct": rel, "blobs": blobs,
    }
    html = open(os.path.join(HERE, "viewer_template.html")).read().replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    open(a.out, "w").write(html)
    print(a.out, round(len(html) / 1e6, 2), "MB", {k: round(len(v) / 1e6, 2) for k, v in blobs.items()})
    print("tracks", [(t["name"], len(t["fc"]), len(t["tr"])) for t in tracks])


if __name__ == "__main__":
    main()
