"""Build viewer.html: decimated car surfaces with high-fidelity and predicted Cp, drag table, accuracy summary.

Inputs (pulled from the GPU box into work/): viewer_npz/<model>_<case>.npz and .vtp.json from postproc.py,
the benchmark's per-case metrics caches (out/<model>*/metrics_cache/*/run_*.json), and cars.json (published Cd).
Drag is integrated by postproc.py from the predicted surface fields: Cd = D / (q A), q = U^2/2 with U = 38.889 m/s,
A = aRef for DrivAerML cars and controls, and the mesh's frontal area for the production cars.
usage: build_viewer.py   (run from this directory; REAL_ONLY=1 leaves out the production cars)
"""
import base64, glob, gzip, json, math, os, numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__)); W = f"{HERE}/work"
cars = json.load(open(f"{HERE}/cars.json"))
pp = [json.load(open(f)) for f in glob.glob(f"{W}/viewer_npz/*.vtp.json")]
Q = 0.5 * 38.889 ** 2
CP_LO, CP_HI = -1.6, 1.0
b64 = lambda a: base64.b64encode(gzip.compress(np.ascontiguousarray(a).tobytes(), 9)).decode()   # gzip, then base64; the page inflates with DecompressionStream
def qcp(p):   # Cp in 0..254, 255 = NaN (the model returned no value there)
    u = np.clip(np.round((p / Q - CP_LO) / (CP_HI - CP_LO) * 254), 0, 254)
    return np.where(np.isfinite(p), u, 255).astype(np.uint8)
MODEL = {"geotransolver": "GeoTransolver", "domino": "DoMINO"}
fin = lambda v: v if v is not None and math.isfinite(v) else None

out = []
for c in cars:
    if c["kind"] == "production" and os.environ.get("REAL_ONLY"): continue
    rs = {r["model"]: r for r in pp if r["case"] == c["case"]}
    npz = {m: np.load(f) for m in MODEL if os.path.exists(f := f"{W}/viewer_npz/{m}_{c['case']}.npz")}
    if not npz: continue
    ref = npz.get("geotransolver", next(iter(npz.values())))
    pts, tri = ref["pts"], ref["tri"]
    lo = pts.min(0); sc = np.maximum(np.ptp(pts, 0), 1e-6) / 65535
    fields = {}
    if c["kind"] != "production": fields["truth"] = b64(qcp(ref["p_true"]))
    for m, z in npz.items():
        p = z["p_pred"] if z is ref else z["p_pred"][cKDTree(z["pts"]).query(pts)[1]]
        fields[m] = b64(qcp(p))
    cd = {m: fin(r["Fx_pred"] / (Q * r.get("aRef", r["A_frontal"]))) for m, r in rs.items()}
    if c["kind"] != "production": cd["truth"] = next(iter(rs.values()))["Cd_hifi"]
    nan = {MODEL[m]: r["nan_frac"] for m, r in rs.items() if r["nan_frac"] > 0}
    note = "; ".join(f"{k} returned NaN on {100 * v:.0f}% of the surface" for k, v in nan.items()) or None
    out.append(dict(c, id=c["case"], geom=dict(pts=b64(np.round((pts - lo) / sc).astype(np.uint16)), tri=b64(tri.astype(np.uint16 if len(pts) < 65536 else np.uint32)), idx16=bool(len(pts) < 65536), lo=lo.tolist(), scale=sc.tolist()),
                    fields=fields, cd=cd, nan_note=note, A_frontal=next(iter(rs.values()))["A_frontal"]))

# accuracy from the benchmark's own metrics (relative drag error, area-weighted pressure L2), per case set
def bench_of(case_ok):
    res = {}
    for m in MODEL:
        per = {}
        for f in glob.glob(f"{W}/out/{m}_surface*/metrics_cache/*/run_*.json"):
            j = json.load(open(f))
            if case_ok(j["case_id"]): per[j["case_id"]] = j["metrics"]
        if per:
            e = [100 * abs(v["drag_error"]) for v in per.values()]
            res[m] = dict(label=MODEL[m], n=len(per), drag_err_mean_pct=float(np.mean(e)), drag_err_max_pct=float(np.max(e)),
                          l2_p_mean=float(np.mean([v["l2_pressure_area_weighted"] for v in per.values()])), cases=sorted(per))
    return res
num = lambda cid: int(cid.split("_")[1])
bench = bench_of(lambda cid: num(cid) < 9000)
# controls: predicted Cd on the wrapped mesh vs the original high-fidelity Cd, i.e. judged exactly like a production car
control = {}
for m in MODEL:
    e = [100 * (c["cd"][m] - c["cd"]["truth"]) / c["cd"]["truth"] for c in out if c["kind"] == "control" and c["cd"].get(m) is not None]
    if e: control[m] = dict(label=MODEL[m], n=len(e), cd_err_mean_pct=float(np.mean(e)), cd_err_abs_max_pct=float(np.max(np.abs(e))))
data = dict(cars=out, bench=dict(models=bench, control=control, note="Relative drag error and area-weighted pressure L2 error against the DrivAerML high-fidelity (HRLES) solution, from NVIDIA's physicsnemo-cfd benchmark metrics, over the 48-car validation split (held out from training)."),
            q=Q, cd_note="DrivAerML and controls: reference is the high-fidelity CFD Cd. Production cars: reference is the manufacturer's published Cd, and the AI value uses this mesh's frontal area. The fan meshes are approximate (smoothed underbody, closed grilles, simplified wheels), so treat those numbers as a stress test, not a validation.",
            caveats=json.load(open(f"{HERE}/caveats.json")), footer=json.load(open(f"{HERE}/results/stamp.json"))["footer"])
html = open(f"{HERE}/viewer_template.html").read().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
open(f"{HERE}/viewer.html", "w").write(html)
json.dump(dict(q=Q, cars=[{k: v for k, v in c.items() if k not in ("geom", "fields")} for c in out], bench=bench, control=control),
          open(f"{HERE}/results/drag.json", "w"), indent=1)
print("viewer.html", round(len(html) / 1e6, 2), "MB;", len(out), "cars")
