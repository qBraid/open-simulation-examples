"""Real superconducting-qubit parameters to set beside the designed transmon.

Sources, in order of preference:
  1. `qbraid devices get <qrn>` for live status (read-only, free).
  2. Calibration snapshots bundled with qiskit-ibm-runtime's fake backends.
     These are real IBM calibrations frozen on the date in `last_update_date`.
     They are NOT live. The qBraid device API does not expose T1/T2/frequency
     today, and live IBM properties need the user's own IBM token.

IBM Heron snapshots (Fez, Marrakesh, Torino) publish T1/T2 but not qubit
frequency or anharmonicity; Eagle snapshots (Sherbrooke, Kyiv) publish all four.

Usage:  python fetch_device_snapshots.py out.json      (needs qiskit-ibm-runtime)
"""

import json
import subprocess
import sys

import numpy as np
from qiskit_ibm_runtime import fake_provider as fp

DEVICES = {  # fake backend -> qBraid QRN
    "FakeFez": "ibm:ibm:qpu:fez",
    "FakeMarrakesh": "ibm:ibm:qpu:marrakesh",
    "FakeTorino": "ibm:ibm:qpu:torino",
    "FakeSherbrooke": "ibm:ibm:qpu:sherbrooke",
    "FakeKyiv": "ibm:ibm:qpu:kyiv",
}


def qbraid_status(qrn):
    try:
        out = subprocess.run(["qbraid", "devices", "get", qrn, "--no-fmt"],
                             capture_output=True, text=True, timeout=60).stdout
        for tok in ("ONLINE", "OFFLINE", "UNAVAILABLE", "RETIRED"):
            if f"'{tok}'" in out:
                return tok
        return "not listed"
    except Exception:
        return "unknown"


def main(out):
    res = []
    for name, qrn in DEVICES.items():
        if not hasattr(fp, name):
            continue
        b = getattr(fp, name)()
        props = b.properties()

        def col(key, scale):
            v = []
            for i in range(b.num_qubits):
                d = props.qubit_property(i)
                if key in d and d[key][0] is not None:
                    v.append(d[key][0] * scale)
            return v

        cols = {"f01_GHz": col("frequency", 1e-9), "anharmonicity_MHz": col("anharmonicity", 1e-6),
                "T1_us": col("T1", 1e6), "T2_us": col("T2", 1e6)}
        stats = {k: ({"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                      "p90": float(np.percentile(v, 90)), "values": [round(x, 4) for x in v]} if v else None)
                 for k, v in cols.items()}
        res.append({"device": name.replace("Fake", ""), "qrn": qrn, "qubits": b.num_qubits,
                    "snapshot_date": str(props.last_update_date)[:10],
                    "qbraid_status_now": qbraid_status(qrn), **stats})
        print(res[-1]["device"], res[-1]["qbraid_status_now"], res[-1]["snapshot_date"],
              {k: (round(v["median"], 3) if v else None) for k, v in stats.items()})
    json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main(sys.argv[1])
