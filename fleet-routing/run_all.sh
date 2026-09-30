#!/usr/bin/env bash
# Reproduce every result in results/ and rebuild the viewer (~10 min on 4 vCPU).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
$PY race.py data/E-n22-k4.txt  --time 20 --bks 375   --out results/e-n22-k4.json
$PY race.py data/A-n32-k5.vrp  --time 60 --bks 784   --out results/a-n32-k5.json
$PY race.py data/X-n101-k25.vrp --time 60 --certify 240 --bks 27591 --out results/x-n101-k25.json
$PY city.py --stops 80 --time 60 --out results/city.json
OMP_NUM_THREADS=1 $PY qaoa_tsp.py
$PY build_viewer.py
