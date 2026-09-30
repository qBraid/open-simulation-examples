#!/usr/bin/env bash
# End to end: Palace eigenmode run -> validation -> Hamiltonian -> dynamics -> viewer.
# Assumes build_palace.sh has installed Palace into $PREFIX and the Palace source
# (for the mesh and the regression reference) is at $BUILD/palace.
set -euo pipefail
PREFIX=${PREFIX:-/tmp/ose-envs/palace}
BUILD=${BUILD:-/tmp/ose-build}
RUNDIR=${RUNDIR:-/tmp/ose-run}
NP=${NP:-3}
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH="$PREFIX/bin:$PATH"
PY="$PREFIX/bin/python"

mkdir -p "$RUNDIR/mesh"
cp "$BUILD/palace/examples/transmon/mesh/transmon.msh2" "$RUNDIR/mesh/"
cp "$HERE/transmon_coarse.json" "$RUNDIR/"
( cd "$RUNDIR" && /usr/bin/time -v palace -np "$NP" transmon_coarse.json > run.log 2> time.log )

cd "$HERE"
POST="$RUNDIR/postpro/transmon_coarse"
"$PY" validate.py "$POST" "$BUILD/palace/test/data/regression/ref/transmon/transmon_coarse" results/validation.json
mkdir -p results/palace_csv && cp "$POST"/*.csv results/palace_csv/
"$PY" derive_hamiltonian.py "$POST" transmon_coarse.json results/hamiltonian.json
"$PY" simulate_dynamics.py results/hamiltonian.json results/dynamics.json
python3 fetch_device_snapshots.py results/device_snapshots.json   # needs qiskit-ibm-runtime
"$PY" -m pip show vtk >/dev/null 2>&1 || "$PY" -m pip install -q vtk
"$PY" build_viewer.py "$POST" results viewer.html "$(cat results/stamp.txt)"
