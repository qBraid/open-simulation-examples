"""Write the analysed qubit + readout-claw region as a GDS mask (layer 1 = metal, 1 um grid).

This is the exact geometry used in the loss analysis (full chip, not the half model). The
Josephson junction and its <1 um leads are drawn as a placeholder bridge on layer 2 (their
fabrication is a separate e-beam step); the meander resonator and feedline are not included.
usage: python export_gds.py '<layout kwargs json>' out.gds
"""
import json, sys
import gdstk
import layouts
from shapely.geometry import box

kw = json.loads(sys.argv[1]); kw["half"] = False
L = layouts.grounded_transmon(**kw)
p = L["params"]
lib = gdstk.Library(unit=1e-6, precision=1e-9)
cell = lib.new_cell("TRANSMON_QUBIT")
metal = L["conductors"]["ground"].union(L["conductors"]["island"]).union(L["conductors"]["claw"])
polys = []
for g in (metal.geoms if hasattr(metal, "geoms") else [metal]):
    outer = gdstk.Polygon(list(g.exterior.coords)[:-1])
    holes = [gdstk.Polygon(list(h.coords)[:-1]) for h in g.interiors]
    polys += gdstk.boolean(outer, holes, "not", layer=1) if holes else [gdstk.Polygon(outer.points, layer=1)]
cell.add(*polys)
jj = box(-1.0, -p["jg"], 1.0, 0.0)  # junction-lead placeholder across the junction gap
cell.add(gdstk.Polygon(list(jj.exterior.coords)[:-1], layer=2))
cell.add(gdstk.Label(json.dumps({k: p[k] for k in ("W", "L", "G", "jg", "wc", "lc", "cg", "ws")}), (0, -p["jg"] - 40), layer=63))
lib.write_gds(sys.argv[2])
print(sys.argv[2], len(polys), "metal polygons; bbox", cell.bounding_box())
