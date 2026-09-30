"""Ahmed body (Ahmed, Ramm & Faltin 1984) as an STL for snappyHexMesh.

Length 1.044 m, width 0.389 m, height 0.288 m, front radius 0.100 m,
slant length 0.222 m at SLANT degrees, ground clearance 0.050 m. Stilts omitted.
Usage: python geometry.py [slant_deg] [out.stl]
"""
import math
import sys
import gmsh

slant = float(sys.argv[1]) if len(sys.argv) > 1 else 35.0
out = sys.argv[2] if len(sys.argv) > 2 else "ahmed.stl"
L, W, H, R, S, G = 1.044, 0.389, 0.288, 0.100, 0.222, 0.050

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
occ = gmsh.model.occ
body = occ.addBox(0, -W / 2, G, L, W, H)
occ.synchronize()
# round the four edges of the front face (x = 0)
front_edges = []
for _, tag in gmsh.model.getEntities(1):
    x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(1, tag)
    if x1 < 1e-6:
        front_edges.append(tag)
body = occ.fillet([body], front_edges, [R])[0][1]
# slant: remove the wedge above the plane through the rear top edge
dx, dz = S * math.cos(math.radians(slant)), S * math.sin(math.radians(slant))
zt = G + H
p1 = occ.addPoint(L - dx, -W, zt)
p2 = occ.addPoint(L + 0.01, -W, zt)
p3 = occ.addPoint(L + 0.01, -W, zt - dz - 0.01 * math.tan(math.radians(slant)))
p4 = occ.addPoint(L - dx, -W, zt + 0.01)
p5 = occ.addPoint(L + 0.01, -W, zt + 0.01)
loop = occ.addCurveLoop([occ.addLine(p1, p3), occ.addLine(p3, p5), occ.addLine(p5, p4), occ.addLine(p4, p1)])
tri = occ.addPlaneSurface([loop])
wedge = occ.extrude([(2, tri)], 0, 2 * W, 0)
wvol = [e for e in wedge if e[0] == 3]
occ.cut([(3, body)], wvol)
occ.synchronize()
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.012)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.004)
gmsh.model.mesh.generate(2)
gmsh.write(out)
print(f"Ahmed body, slant {slant} deg -> {out}")
gmsh.finalize()
