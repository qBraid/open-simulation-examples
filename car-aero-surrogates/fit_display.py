"""Place each production car's display model (an artist's glTF, for looks only) on the mesh the AI was scored on.
Tries the 24 axis-aligned rotations, scales to the simulation mesh's length, aligns front/ground/centre like wrap_mesh.py,
and keeps the rotation whose surface sits closest to the simulation mesh. Writes the 4x4 matrix into cars.json.
Download each model's glb from its Sketchfab page (cars.json display.url; the download API needs a free account token)
as <dir>/<src>.glb. The committed display/*.glb are those files after:
  gltf-transform optimize in.glb out.glb --instance false --compress meshopt --texture-compress webp --texture-size 1024 --simplify-ratio <0.3-1>
usage: fit_display.py <dir>   (SRC_DIR / FAN_DIR as for build_viewer.py; needs trimesh)
"""
import itertools, json, os, sys, numpy as np, trimesh
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__))
B = dict(__file__=f"{HERE}/build_viewer.py"); exec(open(B["__file__"]).read().split("out = []")[0], B)   # reuse its mesh readers and paths
cars = json.load(open(f"{HERE}/cars.json"))
for c in cars:
    if not c.get("display") or not os.path.exists(f := f"{sys.argv[1]}/{c['src']}.glb"): continue
    rid = c["case"][4:]; vs, _ = B["read_stl"](f"{B['SRC']}/{c['src']}.stl")
    vo = B["fit_to"](vs, B["read_stl"](f"{B['FAN']}/run_{rid}/drivaer_{rid}.stl")[0])
    if c.get("rot"): vo = vo * {"x180": [1, -1, -1], "y180": [-1, 1, -1]}[c["rot"]]
    X0, Z0 = B["X0"], B["Z0"]
    vo[:, 0] += X0 - vo[:, 0].min(); vo[:, 1] -= (vo[:, 1].max() + vo[:, 1].min()) / 2; vo[:, 2] += Z0 - vo[:, 2].min()
    vd = np.vstack([g.vertices for g in trimesh.load(f, force="scene").dump()])
    rng = np.random.default_rng(0); so = vo[rng.choice(len(vo), 5000)]; sd = vd[rng.choice(len(vd), 20000)]; To = cKDTree(vo)
    best = None
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3)); R[range(3), perm] = sg
            if np.linalg.det(R) < 0: continue
            full = vd @ R.T; k = np.ptp(vo[:, 0]) / np.ptp(full[:, 0]); full *= k; lo, hi = full.min(0), full.max(0)
            t = np.array([X0 - lo[0], -(hi[1] + lo[1]) / 2, Z0 - lo[2]]); w = sd @ R.T * k + t
            d = np.median(cKDTree(w).query(so)[0]) + np.median(To.query(w)[0])
            if best is None or d < best[0]: best = (d, R * k, t)
    M = np.eye(4); M[:3, :3], M[:3, 3] = best[1], best[2]
    c["display"]["matrix"] = [round(x, 6) for x in M.T.ravel()]   # column-major, as three.js Matrix4.fromArray reads it
    print(c["short"], f"two-way median gap {1000 * best[0]:.0f} mm")
s = json.dumps(cars, separators=(", ", ": ")).replace("}, {", "},\n {")
open(f"{HERE}/cars.json", "w").write(s.replace("[{", "[\n {", 1).replace("}]", "}\n]") + "\n")
