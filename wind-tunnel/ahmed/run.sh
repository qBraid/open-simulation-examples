#!/usr/bin/env bash
# Ahmed body (35 deg slant), half model, k-omega SST, snappyHexMesh + simpleFoam.
# Usage: run.sh <workdir> [nprocs]     (cfd env active)
set -euo pipefail
WORK=$1; NP=${2:-3}
HERE=$(cd "$(dirname "$0")" && pwd)
PSTREAM="$CONDA_PREFIX/lib/mpich-3.3/libPstream.so"   # see cylinder2d/run.sh
rm -rf "$WORK"; mkdir -p "$WORK"; cp -r "$HERE/case/." "$WORK/"; cd "$WORK"
mkdir -p constant/triSurface
python "$HERE/geometry.py" 35 constant/triSurface/ahmed.stl > log.geometry
blockMesh > log.blockMesh
surfaceFeatureExtract > log.surfaceFeatureExtract
snappyHexMesh -overwrite > log.snappyHexMesh
checkMesh > log.checkMesh || true
sed -i "s/numberOfSubdomains .*/numberOfSubdomains $NP; method scotch;/" system/decomposeParDict
decomposePar -force > log.decomposePar
mpirun -np "$NP" -genv LD_PRELOAD "$PSTREAM" potentialFoam -parallel -writephi > log.potentialFoam 2>&1 || true
mpirun -np "$NP" -genv LD_PRELOAD "$PSTREAM" simpleFoam -parallel > log.simpleFoam 2>&1
reconstructPar -latestTime > log.reconstructPar
echo "done: $WORK"
