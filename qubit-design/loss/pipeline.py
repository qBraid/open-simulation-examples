"""One design point end to end: layout -> mesh -> Palace electrostatics -> participation.

usage:
  python pipeline.py table <h_um> <eps_sub> <x0_um> <out.json>          # 2D edge integrals S_i(g)
  python pipeline.py point <rundir> <layout_kwargs_json> <table.json> [--fieldmap] [--np N]
Leaves in <rundir>: meta.json, post/*.csv, participation.json, (fieldmap.npz); deletes the mesh
and ParaView output to keep disk use small.
"""
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def window_for(p):
    return (0.0, -p["jg"] - 250, p["W"] / 2 + p["G"] + p["ws"] + 2 * p["cg"] + p["wc"] + 250,
            p["L"] + p["G"] + 250 + 150)


def point(rundir, kw, table, fieldmap=False, np_=2, edge=1.0):
    import geom3d
    import layouts
    import run_post
    rundir = os.path.abspath(rundir)  # Palace resolves config paths from the run dir
    os.makedirs(rundir, exist_ok=True)
    L = layouts.grounded_transmon(**kw)
    win = window_for(L["params"])
    t0 = time.time()
    attrs, n = geom3d.build(L, f"{rundir}/mesh.msh", edge_size=edge, far_size=250, growth=0.3, window=win)
    cfg, names = geom3d.palace_config(L, attrs, f"{rundir}/mesh.msh", f"{rundir}/post")
    json.dump(cfg, open(f"{rundir}/es.json", "w"), indent=1)
    t1 = time.time()
    with open(f"{rundir}/run.log", "w") as log:
        rc = subprocess.call(["palace", "-np", str(np_), "es.json"], cwd=rundir, stdout=log, stderr=log)
    t2 = time.time()
    if rc:
        raise SystemExit(f"palace failed rc={rc}")
    r = run_post.main(L, f"{rundir}/post", table, 4.0, window=win)
    json.dump(r, open(f"{rundir}/participation.json", "w"), indent=1, default=float)
    if fieldmap:
        run_post.fieldmap(L, f"{rundir}/post", win, out=f"{rundir}/fieldmap.npz")
    t3 = time.time()
    json.dump({"kw": kw, "window": win, "n_tet": n, "names": names,
               "seconds": {"mesh": t1 - t0, "palace": t2 - t1, "post": t3 - t2}},
              open(f"{rundir}/meta.json", "w"), indent=1)
    shutil.rmtree(f"{rundir}/post/paraview", ignore_errors=True)
    os.remove(f"{rundir}/mesh.msh")
    print(json.dumps(r["p"]))


if __name__ == "__main__":
    if sys.argv[1] == "table":
        import stable
        h, eps, x0, out = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
        tab = stable.table(h=h, eps_s=eps, x0=x0, gaps=(2, 3, 6, 10, 20, 30, 50, 100, 200, 500))
        json.dump({str(k): v for k, v in tab.items()}, open(out, "w"), indent=1)
    else:
        a = sys.argv[2:]
        npr = int(a[a.index("--np") + 1]) if "--np" in a else 2
        point(a[0], json.loads(a[1]), a[2], fieldmap="--fieldmap" in a, np_=npr)
