"""Sample vorticity (z) snapshots onto a regular grid for the viewer.

Usage: python frames.py <case_dir> <out.npz>   (run `postProcess -func vorticity` first)
Grid: x in [-2, 14], y in [-3, 3], NX x NY points.
"""
import sys
import numpy as np
import pyvista as pv

case, out = sys.argv[1], sys.argv[2]
NX, NY = 256, 96
open(f"{case}/case.foam", "a").close()
reader = pv.OpenFOAMReader(f"{case}/case.foam")
times = [t for t in reader.time_values if t >= 185 - 1e-9]
grid = pv.ImageData(dimensions=(NX, NY, 1), spacing=(16 / (NX - 1), 6 / (NY - 1), 1), origin=(-2, -3, 0.5))
frames = []
for t in times:
    reader.set_active_time_value(t)
    mesh = reader.read()["internalMesh"]
    s = grid.sample(mesh)
    w = s["vorticity"][:, 2] if "vorticity" in s.array_names else np.zeros(NX * NY)
    w = np.where(s["vtkValidPointMask"] > 0, w, np.nan)
    frames.append(w.reshape(NY, NX))
arr = np.array(frames, dtype=np.float32)
np.savez_compressed(out, w=arr, t=np.array(times), x=(-2, 14), y=(-3, 3))
print(out, arr.shape, float(np.nanmin(arr)), float(np.nanmax(arr)))
