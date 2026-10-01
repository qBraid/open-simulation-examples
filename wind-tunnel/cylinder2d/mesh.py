"""2D cylinder mesh for Re=100 validation (D=1), extruded one cell for OpenFOAM.

Usage: python mesh.py <coarse|fine> <out.msh> [half_width] [inlet_dist]
Domain: x in [-xin, 25] D, y in [-hw, hw] D; hw = 10 and xin = 10 by default (5% blockage). Quad-dominant mesh with a
structured boundary layer on the cylinder, extruded to one hex/prism layer.
"""
import sys
import gmsh

level = sys.argv[1]
out = sys.argv[2]
hw = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
xin = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
XOUT = 25.0
P = {"coarse": dict(n=96, h0=0.02, hwake=0.12, hfar=0.8),
     "fine": dict(n=192, h0=0.01, hwake=0.06, hfar=0.5)}[level]

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("cyl")
occ = gmsh.model.occ
box = occ.addRectangle(-xin, -hw, 0, xin + XOUT, 2 * hw)
disk = occ.addDisk(0, 0, 0, 0.5, 0.5)
surf, _ = occ.cut([(2, box)], [(2, disk)])
occ.synchronize()

curves = gmsh.model.getBoundary(surf, oriented=False)
cyl_curves, edges = [], {}
for dim, tag in curves:
    xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
    if xmax - xmin < 1.01 and ymax - ymin < 1.01:
        cyl_curves.append(tag)
    elif xmax < -xin + 0.01:
        edges["inlet"] = tag
    elif xmin > XOUT - 0.01:
        edges["outlet"] = tag
    else:
        edges.setdefault("sides", []).append(tag)
for c in cyl_curves:
    gmsh.model.mesh.setTransfiniteCurve(c, P["n"] // len(cyl_curves) + 1)

# boundary layer on the cylinder
bl = gmsh.model.mesh.field.add("BoundaryLayer")
gmsh.model.mesh.field.setNumbers(bl, "CurvesList", cyl_curves)
gmsh.model.mesh.field.setNumber(bl, "Size", P["h0"])
gmsh.model.mesh.field.setNumber(bl, "Ratio", 1.15)
gmsh.model.mesh.field.setNumber(bl, "Thickness", 0.25)
gmsh.model.mesh.field.setNumber(bl, "Quads", 1)
gmsh.model.mesh.field.setAsBoundaryLayer(bl)

# refinement: near body and in the wake
dist = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(dist, "CurvesList", cyl_curves)
th = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(th, "InField", dist)
gmsh.model.mesh.field.setNumber(th, "SizeMin", P["hwake"] * 0.6)
gmsh.model.mesh.field.setNumber(th, "SizeMax", P["hfar"])
gmsh.model.mesh.field.setNumber(th, "DistMin", 0.5)
gmsh.model.mesh.field.setNumber(th, "DistMax", 8)
wake = gmsh.model.mesh.field.add("Box")
for k, v in dict(VIn=P["hwake"], VOut=P["hfar"], XMin=-1, XMax=15, YMin=-2.0, YMax=2.0, Thickness=2).items():
    gmsh.model.mesh.field.setNumber(wake, k, v)
mn = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(mn, "FieldsList", [th, wake])
gmsh.model.mesh.field.setAsBackgroundMesh(mn)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.Algorithm", 8)  # frontal-delaunay for quads

ext = gmsh.model.occ.extrude(surf, 0, 0, 1, numElements=[1], recombine=True)
occ.synchronize()
vol = [e[1] for e in ext if e[0] == 3]
# classify extruded surfaces by bounding box
front, back, cyl, inlet, outlet, sides = [], [], [], [], [], []
for dim, tag in gmsh.model.getEntities(2):
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
    if zmax < 1e-6:
        back.append(tag)
    elif zmin > 1 - 1e-6:
        front.append(tag)
    elif xmax - xmin < 1.01 and ymax - ymin < 1.01:
        cyl.append(tag)
    elif xmax < -xin + 0.01:
        inlet.append(tag)
    elif xmin > XOUT - 0.01:
        outlet.append(tag)
    else:
        sides.append(tag)
for name, tags in dict(frontAndBack=front + back, cylinder=cyl, inlet=inlet, outlet=outlet, sides=sides).items():
    gmsh.model.addPhysicalGroup(2, tags, name=name)
gmsh.model.addPhysicalGroup(3, vol, name="fluid")
gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write(out)
n = len(gmsh.model.mesh.getElementsByType(5)[0]) + len(gmsh.model.mesh.getElementsByType(6)[0])
print(f"{level}: {n} volume cells -> {out}")
gmsh.finalize()
