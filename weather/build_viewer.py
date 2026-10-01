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
    "tcwv": dict(label="Total column water vapour", units="kg/m²", lo=0, hi=70, cmap="ice", err=12, alpha_lo=0.3),
    "t2m": dict(label="2 m temperature", units="°C", lo=-50, hi=45, cmap="thermal", err=5),
    "msl": dict(label="Sea-level pressure", units="hPa", lo=960, hi=1045, cmap="deep_r", err=8),
}
CONV = {"tcwv": lambda a: a, "t2m": lambda a: a - 273.15, "msl": lambda a: a / 100.0}
WIND_SCALE = 0.5  # m/s per int8 step
LABEL = {"sfno": "SFNO-small", "fcn3": "FourCastNet 3"}
FULL = {"sfno": "NVIDIA SFNO (sfno_73ch_small)", "fcn3": "NVIDIA FourCastNet 3 (FCN3)"}
STORMS = [  # name, approx position at init (lon, lat), search radius deg
    ("Hurricane Laura", -72.0, 18.0, 6.0),
    ("Typhoon Bavi", 125.5, 24.0, 6.0),
]


def b64z(arr):
    return np.ascontiguousarray(arr).tobytes()  # raw bytes; packed and deflated once in pack_blobs


def pack_blobs(blobs):
    """One deflate stream for every array (one DecompressionStream in the page). Offsets are
    2-byte aligned so int16 views work."""
    buf, index, off = bytearray(), {}, 0
    for k, b in blobs.items():
        if off % 2: buf += b"\0"; off += 1
        index[k] = [off, len(b)]; buf += b; off += len(b)
    return base64.b64encode(zlib.compress(bytes(buf), 9)).decode(), index


def b64z_rows(arr):
    """8-bit fields: PNG-style 'sub' filter along longitude (byte deltas mod 256), then deflate."""
    u = np.ascontiguousarray(arr).view(np.uint8)
    d = np.diff(u, axis=-1, prepend=np.zeros(u.shape[:-1] + (1,), np.uint8)).astype(np.uint8)
    return b64z(d)


def b64z_pts(p):
    """int16 (lon, lat) pairs: delta-encode each component along the stream, then deflate."""
    p = np.ascontiguousarray(p, dtype=np.int16).reshape(-1, 2)
    d = np.diff(p, axis=0, prepend=np.zeros((1, 2), np.int16)).astype(np.int16)
    return b64z(d)


def down(a):  # 0.25 deg (721 x 1440, lon 0..359.75) -> 1 deg (181 x 360)
    return a[::4, ::4]


def down2(a):  # 0.25 deg -> 2 deg (91 x 180) for the wind particles
    return a[::8, ::8]


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
            line = line[:: max(1, len(line) // 110)]  # cap points per line (1 deg grid: smooth enough)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--models", required=True, help="headline first: fcn3=res/fcn3_viz.npz,sfno=res/sfno_viz.npz")
    ap.add_argument("--out", default=os.path.join(HERE, "viewer.html")); ap.add_argument("--init", default="2020-08-24T00:00")
    a = ap.parse_args()
    specs = [kv.split("=", 1) for kv in a.models.split(",")]
    names = [m for m, _ in specs]; headline = names[0]
    step = 6 if len(specs) == 1 else 12            # keep the page under ~5 MB with two models
    leads = list(range(0, 121, step)); tleads = list(range(0, 121, 12))
    blobs, vmeta, z5, models, tracks_fc = {}, {}, {}, {}, {}
    levels = list(range(4800, 6001, 120))
    Z0 = None
    for m, path in specs:
        Z = np.load(path)
        if Z0 is None: Z0 = Z
        for v, mm in VMETA.items():
            blobs[f"fc_{m}_{v}"] = b64z_rows(np.stack([q8(CONV[v](down(Z[f"{v}_{h:03d}"])), mm["lo"], mm["hi"]) for h in leads])); vmeta[v] = mm
        for c in ("u850", "v850"):
            blobs[f"fc_{m}_{c}"] = b64z_rows(np.stack([np.clip(np.round(down2(Z[f"{c}_{h:03d}"]) / WIND_SCALE), -127, 127).astype(np.int8) for h in leads]))
        metas, allp = [], []
        for h in leads:
            m_, p_ = contours(down(Z[f"z500_{h:03d}"]) / 9.80665, levels); metas.append(m_); allp.append(p_)
        z5[f"fc_{m}"] = metas; blobs[f"z5_fc_{m}_pts"] = b64z_pts(np.concatenate(allp))
        lat, lon = Z["lat"], Z["lon"]
        tracks_fc[m] = [track([Z[f"msl_{h:03d}"] for h in range(0, 121, 6)], lo0, la0, rad, lat, lon) for _, lo0, la0, rad in STORMS]
        S = json.load(open(os.path.join(a.results, f"{m}_scores.json")))
        models[m] = {"label": FULL[m], "short": LABEL[m], "gpu": S.get("gpu", "NVIDIA GPU"), "step_s": float(np.mean(S["step_s"])), "n": len(S["inits"])}
        Z = None
    for v, mm in VMETA.items():
        blobs[f"tr_{v}"] = b64z_rows(np.stack([q8(CONV[v](down(Z0[f"era5_{v}_{h:03d}"])), mm["lo"], mm["hi"]) for h in tleads]))
    for c in ("u850", "v850"):
        blobs[f"tr_{c}"] = b64z_rows(np.stack([np.clip(np.round(down2(Z0[f"era5_{c}_{h:03d}"]) / WIND_SCALE), -127, 127).astype(np.int8) for h in tleads]))
    metas, allp = [], []
    for h in tleads:
        m_, p_ = contours(down(Z0[f"era5_z500_{h:03d}"]) / 9.80665, levels); metas.append(m_); allp.append(p_)
    z5["tr"] = metas; blobs["z5_tr_pts"] = b64z_pts(np.concatenate(allp))
    land_pts, land_lens = pack(os.path.join(HERE, "data", "land-110m.json" if len(specs) > 1 else "land-50m.json"))
    blobs["land_pts"] = b64z_pts(land_pts)
    lat, lon = Z0["lat"], Z0["lon"]
    tracks = []
    for k, (name, lo0, la0, rad) in enumerate(STORMS):
        tr = track([Z0[f"era5_msl_{h:03d}"] for h in tleads], lo0, la0, rad, lat, lon)
        tracks.append({"name": name, "fc": {m: tracks_fc[m][k] for m in names}, "tr": tr})

    H = json.load(open(os.path.join(a.results, "hres_scores.json")))
    R = json.load(open(os.path.join(a.results, "wb2_ref_2020.json")))
    SC = {m: json.load(open(os.path.join(a.results, f"{m}_scores.json"))) for m in names}
    init = datetime.fromisoformat(a.init)

    def series(var):
        k = ["24", "72", "120"]; hp = H["rmse"][var]; out = []
        for i, m in enumerate(names):
            sf = SC[m]["rmse"][var]
            out.append({"label": f"{LABEL[m]} ({len(SC[m]['inits'])} starts)", "color": "--c-model" if i == 0 else "--c-model2", "bold": i == 0, "dash": i > 0,
                        "vals": [float(np.mean(sf[h])) for h in k],
                        **({"band": [[float(np.min(sf[h])), float(np.max(sf[h]))] for h in k]} if i == 0 else {})})
        out += [
            {"label": f"IFS HRES, same {len(H['inits'])} starts", "color": "--c-hres", "vals": [float(np.mean(hp[h])) for h in k]},
            {"label": "IFS HRES, WB2 2020 (730 starts)", "color": "--c-hres", "dash": True, "vals": [R["hres"][var].get(h) for h in k]},
            {"label": "GraphCast, WB2 2020", "color": "--c-gc", "dash": True, "vals": [R["graphcast"][var].get(h) for h in k]},
            {"label": "Pangu, WB2 2020", "color": "--c-pangu", "dash": True, "vals": [R["pangu"][var].get(h) for h in k]},
        ]
        return out

    verdict = json.load(open(os.path.join(a.results, "verdict.json"))) if os.path.exists(os.path.join(a.results, "verdict.json")) else {}
    payload = {
        "models": models, "headline": headline, "fc_step": step, "init": init.isoformat()[:16],
        "grid": {"nx": 360, "ny": 181, "wnx": 180, "wny": 91}, "leads": leads, "truth_leads": tleads,
        "vars": list(VMETA), "vmeta": vmeta, "wind_scale": WIND_SCALE,
        "cmaps": {"ice": lut("ice"), "thermal": lut("thermal"), "deep_r": lut("deep_r"), "diverging": lut("balance")},
        **{f"z5_{k}": v for k, v in z5.items()}, "land_lens": land_lens.tolist(), "tracks": tracks,
        "scores": {v: series(v) for v in ("z500", "t850", "t2m")},
        "score_labels": {"z500": "z500 (m²/s²)", "t850": "T850 (K)", "t2m": "T2m (K)"},
        "verdict_html": verdict.get("html", ""), "stamp_html": verdict.get("stamp_html", ""), "tour": verdict.get("tour", []),
    }
    sizes = {k: round(len(v) / 1e6, 2) for k, v in blobs.items()}
    payload["pack"], payload["index"] = pack_blobs(blobs)
    html = open(os.path.join(HERE, "viewer_template.html")).read().replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    open(a.out, "w").write(html)
    print(a.out, round(len(html) / 1e6, 2), "MB (raw MB per array:", sizes, ")")
    print("tracks", [(t["name"], {m: len(v) for m, v in t["fc"].items()}, len(t["tr"])) for t in tracks])


if __name__ == "__main__":
    main()
