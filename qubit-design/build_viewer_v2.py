"""Build viewer.html (v2): chip + eigenmode fields, loss map + design evolution, panels.

Inputs (all small, committed under results/):
  results/v2/viewer_data/*.json   chip meshes + quantized eigenmode fields (from build_chip below)
  results/v2/*_participation.json edge participation densities (electrostatic half models)
  results/v2/design_*.json        Hamiltonian + T1 budgets
  results/v2/sweep.json           geometry sweep
  results/v2/validation_wang2015.json
  results/device_snapshots.json   IBM calibration snapshots
Usage:  python build_viewer_v2.py        (writes viewer.html)
        python build_viewer_v2.py chip <palace_postpro_dir> <out.json> <mode_freqs_json>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "loss"))


def build_chip(postpro, out, info):
    """Metal-surface mesh with |E| for each eigenmode from Palace boundary output."""
    import glob
    sys.path.insert(0, HERE)
    from build_viewer import quantize, read_cycle
    cycles = sorted(glob.glob(f"{postpro}/paraview/eigenmode_boundary/Cycle*/data.pvtu"))[:2]
    geom, fields = None, []
    for k, c in enumerate(cycles):
        pts, tris, mags = read_cycle(c)
        used = np.unique(tris)
        key = np.round(pts[used], 6)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        remap = np.full(len(pts), -1)
        remap[used] = inv.ravel()
        idx = remap[tris]
        if geom is None:
            geom = {"pos": np.round(uniq, 2).ravel().tolist(), "idx": idx.ravel().tolist(), "n": int(len(uniq))}
        v = np.zeros(len(uniq))
        np.maximum.at(v, inv.ravel(), mags["E"][used])
        fields.append(quantize(v).tolist())
    P = np.array(geom["pos"]).reshape(-1, 3)
    lo, hi = P.min(0), P.max(0)
    geom.update(fields=fields, lo=lo.tolist(), hi=hi.tolist(), info=info)
    json.dump(geom, open(out, "w"), separators=(",", ":"))
    print(out, geom["n"], "nodes", len(geom["idx"]) // 3, "tris")


def layout_polys(kw):
    """Qubit-region polygons (full, unmirrored) with a fixed vertex structure for morphing."""
    import layouts
    from shapely.geometry import box
    from shapely.ops import unary_union
    L = layouts.grounded_transmon(**{**kw, "half": False})
    p = L["params"]
    wg = p["W"] + 2 * p["G"]
    yt = p["L"] + p["G"]
    ws, cg, wc, lc = p["ws"], p["cg"], p["wc"], p["lc"]
    trench = box(-wg / 2, -p["jg"], wg / 2, yt)
    h3x0 = wg / 2 + ws
    holes = unary_union([
        box(-(wg / 2 + ws + 2 * cg + wc), yt + ws, wg / 2 + ws + 2 * cg + wc, yt + ws + wc + 2 * cg),
        box(-(h3x0 + 2 * cg + wc), yt + ws - (lc + cg), -h3x0, yt + ws),
        box(h3x0, yt + ws - (lc + cg), h3x0 + 2 * cg + wc, yt + ws),
        box(-11, yt + ws + wc + 2 * cg - 0.01, 11, yt + ws + wc + 2 * cg + 150)])
    def ring(g):
        g = g.simplify(0.01)
        return [[round(x, 2), round(y, 2)] for x, y in list(g.exterior.coords)[:-1]]
    return {"island": ring(L["conductors"]["island"]), "claw": ring(L["conductors"]["claw"]),
            "trench": ring(trench), "clawhole": ring(holes), "params": p}


def mirror_density(ed):
    ed = np.array(ed)
    m = ed.copy()
    m[:, 0] *= -1
    keep = ed[:, 0] > 1e-3
    return np.vstack([ed, m[keep]])


def pack_density(ed):
    ed = mirror_density(ed)
    d = np.log10(np.maximum(ed[:, 2], 1e-30))
    return {"x": np.round(ed[:, 0], 1).tolist(), "y": np.round(ed[:, 1], 1).tolist(),
            "q": np.round(d, 3).tolist()}


def main():
    R = os.path.join(HERE, "results")
    V = os.path.join(R, "v2")
    data = {"stamp": open(os.path.join(V, "stamp.txt")).read().strip()}
    data["chip"] = {k: json.load(open(os.path.join(V, "viewer_data", f"chip_{k}.json")))
                    for k in ("v1", "final") if os.path.exists(os.path.join(V, "viewer_data", f"chip_{k}.json"))}
    designs = json.load(open(os.path.join(V, "designs.json")))
    for d in designs:
        d["polys"] = layout_polys(d["kw"])
        if d.get("participation"):
            pj = json.load(open(os.path.join(V, d["participation"])))
            if "edge_density" in pj:
                d["density"] = pack_density(pj["edge_density"])
    data["designs"] = designs
    data["sweep"] = json.load(open(os.path.join(V, "sweep.json")))
    fm = os.path.join(V, "final_fieldmap.npz")
    if os.path.exists(fm):
        import base64
        z = np.load(fm)
        E = np.log10(np.maximum(z["Emag"], 1e-30))
        hi = np.percentile(E, 99.8); lo = hi - 2.5
        q = np.clip((E - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        q = np.concatenate([q[:, ::-1], q], axis=1)  # mirror the half model about x = 0
        data["fieldmap"] = {"nx": int(q.shape[1]), "ny": int(q.shape[0]),
                            "x0": float(-z["x"][-1]), "x1": float(z["x"][-1]),
                            "y0": float(z["y"][0]), "y1": float(z["y"][-1]),
                            "b64": base64.b64encode(q.tobytes()).decode()}
    data["validation"] = json.load(open(os.path.join(V, "validation_wang2015.json")))
    devices = json.load(open(os.path.join(R, "device_snapshots.json")))
    for d in devices:
        for k in ("f01_GHz", "anharmonicity_MHz", "T1_us", "T2_us"):
            if d.get(k):
                d[k].pop("values", None)
    data["devices"] = devices
    tpl = open(os.path.join(HERE, "viewer_template_v2.html")).read()
    html = tpl.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    open(os.path.join(HERE, "viewer.html"), "w").write(html)
    print(f"viewer.html: {len(html) / 1e6:.2f} MB")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "chip":
        build_chip(sys.argv[2], sys.argv[3], json.loads(sys.argv[4]))
    else:
        main()
