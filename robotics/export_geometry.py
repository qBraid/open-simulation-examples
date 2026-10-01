"""Export the Go1 scene geometry (visual meshes + primitives) for the three.js viewer.

Each geom is written with its parent body index and its pose relative to that
body, so the viewer can place it from the recorded body world poses
(xpos, xquat). Meshes are vertex-clustered to keep the page small.
"""
import json
import sys

import mujoco
import numpy as np
from mujoco_playground import registry

env = registry.load(sys.argv[1] if len(sys.argv) > 1 else "Go1JoystickFlatTerrain")
m = env.mj_model
GRID = float(sys.argv[2]) if len(sys.argv) > 2 else 0.004  # clustering cell (m)


def cluster(v, f, cell):
    if cell <= 0:
        return v, f
    key = np.floor(v / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    nv = inv.max() + 1
    acc = np.zeros((nv, 3)); cnt = np.zeros(nv)
    np.add.at(acc, inv, v); np.add.at(cnt, inv, 1)
    v2 = acc / cnt[:, None]
    f2 = inv[f]
    keep = (f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 0] != f2[:, 2])
    return v2, f2[keep]


geoms, meshes, mesh_index = [], [], {}
tot_v = tot_f = 0
for g in range(m.ngeom):
    gtype = int(m.geom_type[g])
    body = int(m.geom_bodyid[g])
    group = int(m.geom_group[g])
    rgba = m.geom_rgba[g].tolist()
    if m.geom_matid[g] >= 0:
        rgba = m.mat_rgba[m.geom_matid[g]].tolist()
    if rgba[3] == 0:
        continue
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    rec = {"name": name, "body": body, "group": group, "type": gtype,
           "pos": m.geom_pos[g].round(5).tolist(), "quat": m.geom_quat[g].round(6).tolist(),
           "size": m.geom_size[g].round(5).tolist(), "rgba": [round(x, 3) for x in rgba]}
    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[g])
        if mid not in mesh_index:
            va, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            fa, nf = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            v = m.mesh_vert[va:va + nv].astype(np.float64)
            f = m.mesh_face[fa:fa + nf].astype(np.int64)
            v2, f2 = cluster(v, f, GRID)
            mesh_index[mid] = len(meshes)
            meshes.append({"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid),
                           "v": np.round(v2, 4).ravel().tolist(), "f": f2.ravel().tolist()})
            tot_v += len(v2); tot_f += len(f2)
        rec["mesh"] = mesh_index[mid]
    geoms.append(rec)

bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}" for b in range(m.nbody)]
json.dump({"bodies": bodies, "geoms": geoms, "meshes": meshes, "cell_m": GRID},
          open("geometry.json", "w"))
print(f"geoms={len(geoms)} meshes={len(meshes)} verts={tot_v} faces={tot_f} groups={sorted(set(x['group'] for x in geoms))}")
for x in geoms:
    print(x["name"], x["body"], x["group"], x["type"], x.get("mesh"), x["rgba"])
