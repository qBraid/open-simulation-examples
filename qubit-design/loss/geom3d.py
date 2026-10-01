"""Planar superconducting layout -> 3D Palace electrostatic model (mesh + config).

Metal is modelled as zero-thickness sheets on the substrate top face (z = 0), the
standard global-model assumption; the finite film thickness is restored by the local 2D
edge model (edge2d.py). The chip sits in a grounded metal enclosure with vacuum around
it (a 3D-cavity or package). Units: micrometres (Palace L0 = 1e-6).

A layout is a dict:
    {"conductors": {name: shapely (Multi)Polygon, ...},   # each becomes a Palace terminal
     "ground": name or None,                                # conductor tied to the enclosure
     "chip": (xmin, ymin, xmax, ymax), "t_sub": um, "eps_sub": float,
     "gap_below": um, "gap_above": um, "gap_side": um}
"""
import json

import gmsh
import numpy as np
from shapely.geometry import MultiPolygon, Polygon


def _polys(g):
    return list(g.geoms) if isinstance(g, MultiPolygon) else [g]


def _occ_polygon(occ, poly, z=0.0):
    def loop(coords):
        pts = [occ.addPoint(x, y, z) for x, y in list(coords)[:-1]]
        lines = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        return occ.addCurveLoop(lines)

    loops = [loop(poly.exterior.coords)] + [loop(r.coords) for r in poly.interiors]
    return occ.addPlaneSurface(loops)


def build(layout, mesh_path, edge_size=1.0, far_size=200.0, growth=0.25, window=None):
    """Mesh the layout. edge_size: element size on metal edges. window: optional
    (xmin, ymin, xmax, ymax) where edges are refined (default: whole chip)."""
    x0, y0, x1, y1 = layout["chip"]
    ts, gb, ga, gs = layout["t_sub"], layout["gap_below"], layout["gap_above"], layout["gap_side"]
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("chip")
    occ = gmsh.model.occ
    half = layout.get("half_x", False)
    bx0 = x0 if half else x0 - gs
    box = occ.addBox(bx0, y0 - gs, -ts - gb, (x1 - bx0) + gs, (y1 - y0) + 2 * gs, ts + gb + ga)
    sub = occ.addBox(x0, y0, -ts, x1 - x0, y1 - y0, ts)
    metal = {}
    for name, geom in layout["conductors"].items():
        metal[name] = [_occ_polygon(occ, p) for p in _polys(geom)]
    tools = [(2, s) for v in metal.values() for s in v]
    out, outmap = occ.fragment([(3, box), (3, sub)], tools)
    occ.synchronize()
    # map original metal surfaces -> fragment results
    n_vol = 2
    metal_frag = {}
    i = n_vol
    for name, surfs in metal.items():
        tags = []
        for _ in surfs:
            tags += [t for d, t in outmap[i] if d == 2]
            i += 1
        metal_frag[name] = tags
    vols = gmsh.model.getEntities(3)
    sub_v, air_v = [], []
    for _, v in vols:
        bb = gmsh.model.getBoundingBox(3, v)
        inside = (bb[0] >= x0 - 1e-6 and bb[3] <= x1 + 1e-6 and bb[1] >= y0 - 1e-6
                  and bb[4] <= y1 + 1e-6 and bb[2] >= -ts - 1e-6 and bb[5] <= 1e-6)
        (sub_v if inside else air_v).append(v)
    attrs = {"substrate": 1, "vacuum": 2}
    gmsh.model.addPhysicalGroup(3, sub_v, 1, "substrate")
    gmsh.model.addPhysicalGroup(3, air_v, 2, "vacuum")
    # enclosure faces
    outer = []
    xb0, yb0, zb0 = (x0 - gs if not half else -1e30), y0 - gs, -ts - gb  # half model: x=x0 face is a symmetry plane
    xb1, yb1, zb1 = x1 + gs, y1 + gs, ga
    for _, s in gmsh.model.getEntities(2):
        c = occ.getCenterOfMass(2, s)
        if (abs(c[0] - xb0) < 1e-6 or abs(c[0] - xb1) < 1e-6 or abs(c[1] - yb0) < 1e-6
                or abs(c[1] - yb1) < 1e-6 or abs(c[2] - zb0) < 1e-6 or abs(c[2] - zb1) < 1e-6):
            outer.append(s)
    tag = 3
    gmsh.model.addPhysicalGroup(2, outer, tag, "enclosure")
    attrs["enclosure"] = tag
    for name, tags in metal_frag.items():
        tag += 1
        gmsh.model.addPhysicalGroup(2, tags, tag, name)
        attrs[name] = tag
    # refinement on metal edges
    curves = []
    for tags in metal_frag.values():
        for s in tags:
            for d, c in gmsh.model.getBoundary([(2, s)], oriented=False):
                curves.append(c)
    curves = sorted(set(curves))
    if window is not None:
        keep = []
        for c in curves:
            bb = gmsh.model.getBoundingBox(1, c)
            if bb[3] >= window[0] and bb[0] <= window[2] and bb[4] >= window[1] and bb[1] <= window[3]:
                keep.append(c)
        curves = keep
    f1 = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f1, "CurvesList", curves)
    gmsh.model.mesh.field.setNumber(f1, "Sampling", 400)
    f2 = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(f2, "F", f"Min({far_size}, {edge_size} + {growth}*F{f1})")
    gmsh.model.mesh.field.setAsBackgroundMesh(f2)
    for k in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
        gmsh.option.setNumber(f"Mesh.{k}", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT, fast and parallel
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.model.mesh.generate(3)
    n_tet = sum(len(e) for e in gmsh.model.mesh.getElements(3)[1])
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(mesh_path)
    gmsh.finalize()
    return attrs, n_tet


def palace_config(layout, attrs, mesh_path, out_dir, order=2):
    names = [n for n in layout["conductors"] if n != layout.get("ground")]
    ground = [attrs["enclosure"]] + ([attrs[layout["ground"]]] if layout.get("ground") else [])
    return {
        "Problem": {"Type": "Electrostatic", "Verbose": 1, "Output": out_dir,
                    "OutputFormats": {"Paraview": True, "GridFunction": False}},
        "Model": {"Mesh": mesh_path, "L0": 1e-6},
        "Domains": {"Materials": [
            {"Attributes": [attrs["substrate"]], "Permittivity": layout["eps_sub"]},
            {"Attributes": [attrs["vacuum"]], "Permittivity": 1.0}],
            "Postprocessing": {"Energy": [{"Index": 1, "Attributes": [attrs["substrate"]]},
                                          {"Index": 2, "Attributes": [attrs["vacuum"]]}]}},
        "Boundaries": {"Ground": {"Attributes": ground},
                       "Terminal": [{"Index": i + 1, "Attributes": [attrs[n]]} for i, n in enumerate(names)]},
        "Solver": {"Order": order, "Electrostatic": {"Save": len(names)},
                   "Linear": {"Type": "BoomerAMG", "KSPType": "CG", "Tol": 1e-10, "MaxIts": 400,
                              "EstimatorTol": 1e-2, "EstimatorMaxIts": 20, "EstimatorMG": False}},
    }, names


if __name__ == "__main__":
    import sys
    print(json.dumps({"ok": True}))
