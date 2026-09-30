"""Build the self-contained three.js viewer (viewer.html) from a Palace run.

Reads Palace's ParaView boundary output (Lagrange triangles with E, J_s per
point), keeps the metal surfaces (mesh attribute 5 = PEC, 4 = junction),
reduces each high-order triangle to its three corner nodes, merges nodes
shared between MPI partitions, and quantizes log10 field magnitudes to 0..255.
The Hamiltonian, dynamics, validation and device snapshots from results/ are
inlined, so the page needs nothing but the three.js CDN.

Usage:  python build_viewer.py <run_postpro> results/ viewer.html "<stamp text>"
"""

import glob
import json
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

METAL = {4, 5}
LABELS = {1: "qubit", 2: "resonator"}


def read_cycle(path):
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(path)
    r.Update()
    g = r.GetOutput()
    pts = vtk_to_numpy(g.GetPoints().GetData()).astype(float)
    attr = vtk_to_numpy(g.GetCellData().GetArray("attribute"))
    conn = vtk_to_numpy(g.GetCells().GetConnectivityArray())
    offs = vtk_to_numpy(g.GetCells().GetOffsetsArray())
    tris = np.array([conn[offs[i]:offs[i] + 3] for i in range(len(offs) - 1) if int(attr[i]) in METAL])

    def mag(name):
        re = vtk_to_numpy(g.GetPointData().GetArray(name + "_real"))
        im = vtk_to_numpy(g.GetPointData().GetArray(name + "_imag"))
        return np.sqrt((re ** 2).sum(1) + (im ** 2).sum(1))

    return pts, tris, {"E": mag("E"), "J_s": mag("J_s")}


def quantize(v, decades=3.0):
    lv = np.log10(np.maximum(v, 1e-300))
    hi = np.percentile(lv, 99.7)
    return np.clip((lv - (hi - decades)) / decades * 255, 0, 255).round().astype(int)


def main(postpro, results, out, stamp):
    cycles = sorted(glob.glob(f"{postpro}/paraview/eigenmode_boundary/Cycle*/data.pvtu"))
    ham = json.load(open(f"{results}/hamiltonian.json"))
    geom, fields = None, []
    for k, c in enumerate(cycles, start=1):
        pts, tris, mags = read_cycle(c)
        used = np.unique(tris)
        # merge nodes duplicated across partitions (identical coordinates)
        key = np.round(pts[used], 6)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        remap = np.full(len(pts), -1)
        remap[used] = inv.ravel()
        idx = remap[tris]
        if geom is None:
            geom = {"pos": np.round(uniq, 2).ravel().tolist(), "idx": idx.ravel().tolist(), "n": len(uniq)}
        f_ghz = ham["inputs"]["mode_freqs_linear_GHz"][k - 1]
        p = ham["inputs"]["junction_participation"][k - 1]
        for name, label in (("E", "|E|"), ("J_s", "surface current |J_s|")):
            v = np.zeros(len(uniq))
            np.maximum.at(v, inv.ravel(), mags[name][used])
            fields.append({"label": f"{LABELS.get(k, k)} {label}",
                           "info": f"mode {k}: {f_ghz:.4f} GHz (linear), junction participation {p:.4f}",
                           "v": quantize(v).tolist()})
    P = np.array(geom["pos"]).reshape(-1, 3)
    lo, hi = P.min(0), P.max(0)
    geom.update(fields=fields, center=((lo + hi) / 2)[:2].tolist(),
                substrate=[float(hi[0] - lo[0]), float(hi[1] - lo[1]), 525.0], scale=4.0 / float(hi[0] - lo[0]))
    devices = json.load(open(f"{results}/device_snapshots.json"))
    for d in devices:  # keep the page small: medians/percentiles only
        for k in ("f01_GHz", "anharmonicity_MHz", "T1_us", "T2_us"):
            if d.get(k):
                d[k].pop("values", None)
    data = {
        "geom": geom, "ham": ham, "dyn": json.load(open(f"{results}/dynamics.json")),
        "validation": json.load(open(f"{results}/validation.json")), "devices": devices, "stamp": stamp,
        "devNote": ("Medians with 10th–90th percentile across each device's qubits, from the dated calibration "
                    "snapshots in qiskit-ibm-runtime (not live data). Heron devices (Fez, Marrakesh, Torino) do not publish "
                    "frequency or anharmonicity (n/p). Comparable: f01, α and E_J/E_C. <b>Not comparable: T1.</b> "
                    "*The design's T1 is a loss-model bound set by the example's assumed sapphire loss tangent (3e-5 to 8.6e-5, "
                    "deliberately pessimistic), while device T1 reflects fabrication, TLS defects and packaging."),
    }
    dyn = data["dyn"]  # thin the traces
    dyn["ramsey"] = {k: (v[::4] if isinstance(v, list) else v) for k, v in dyn["ramsey"].items()}
    dyn["readout"] = {k: ([round(x, 7) for x in v[::2]] if isinstance(v, list) else v) for k, v in dyn["readout"].items()}
    dyn["rabi"] = {k: ([round(x, 5) for x in v] if isinstance(v, list) else v) for k, v in dyn["rabi"].items()}
    tpl = open("viewer_template.html").read()
    html = tpl.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    open(out, "w").write(html)
    print(f"{out}: {len(html) / 1e6:.2f} MB, {geom['n']} nodes, {len(geom['idx']) // 3} triangles, {len(fields)} fields")


if __name__ == "__main__":
    main(*sys.argv[1:5])
