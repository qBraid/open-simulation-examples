"""Sample the recorded leg of the cylinder run for the v2 viewer.

Usage: python frames_v2.py <case_dir> <out.npz>   (run `postProcess -func vorticity -time 'T1:'` first)
Vorticity on a 256 x 96 grid (texture and relief), velocity on a 128 x 48 grid (dye particles),
both over x in [-2, 14] D, y in [-3, 3] D, plus the Cd/Cl trace of the recorded leg.
"""
import glob
import sys
import numpy as np
import pyvista as pv

case, out = sys.argv[1], sys.argv[2]
X, Y = (-2.0, 14.0), (-3.0, 3.0)


def grid(nx, ny):
    return pv.ImageData(dimensions=(nx, ny, 1), spacing=((X[1] - X[0]) / (nx - 1), (Y[1] - Y[0]) / (ny - 1), 1),
                        origin=(X[0], Y[0], 0.5))


open(f"{case}/case.foam", "a").close()
reader = pv.OpenFOAMReader(f"{case}/case.foam")
tv = [t for t in reader.time_values if t > 0]
t1 = max(tv) - 15.0
times = [t for t in tv if t >= t1 - 1e-9]
gw, gu = grid(256, 96), grid(128, 48)
W, U = [], []
for t in times:
    reader.set_active_time_value(t)
    mesh = reader.read()["internalMesh"]
    s = gw.sample(mesh)
    w = s["vorticity"][:, 2] if "vorticity" in s.array_names else np.zeros(256 * 96)
    W.append(np.where(s["vtkValidPointMask"] > 0, w, np.nan).reshape(96, 256))
    s = gu.sample(mesh.cell_data_to_point_data())
    u = np.where(s["vtkValidPointMask"][:, None] > 0, s["U"][:, :2], 0.0)
    U.append(u.reshape(48, 128, 2))
rows = [np.loadtxt(f, comments="#", usecols=(0, 1, 4))
        for f in sorted(glob.glob(f"{case}/postProcessing/forceCoeffs/*/coefficient.dat"))]
d = np.vstack(rows)
d = d[np.argsort(d[:, 0])]
m = d[:, 0] >= t1 - 1e-9
np.savez_compressed(out, w=np.array(W, np.float32), u=np.array(U, np.float32), t=np.array(times),
                    force=d[m][:: max(1, int(m.sum() // 1500))].astype(np.float32), x=X, y=Y)
print(out, len(times), "frames")
