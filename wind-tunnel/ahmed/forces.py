"""Mean Cd/Cl of the Ahmed half model over the last N iterations. Usage: forces.py <case> [N]"""
import glob, json, sys
import numpy as np
case = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
d = np.vstack([np.loadtxt(f, comments="#", usecols=(0, 1, 4))
               for f in sorted(glob.glob(f"{case}/postProcessing/forceCoeffs/*/coefficient.dat"))])
it, cd, cl = d[-N:].T
print(json.dumps(dict(iterations=int(d[-1, 0]), Cd_mean=round(float(cd.mean()), 4),
                      Cd_std=round(float(cd.std()), 4), Cl_mean=round(float(cl.mean()), 4), window=N)))
