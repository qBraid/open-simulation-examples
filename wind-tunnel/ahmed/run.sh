#!/usr/bin/env bash
# Ahmed body, half model, k-omega SST, snappyHexMesh + simpleFoam.
# Usage: run.sh <workdir> [nprocs]                      (cfd env active)
# Env:   SLANT=25|35 (deg, default 35)   LEVEL=coarse|medium|fine (default medium)
#        GROUND=fixed|moving (default fixed: no-slip floor, as in the wind tunnel)
#        ITERS (default 1000)   RESUME=1 skips meshing and continues from the latest write
# The mesh ladder scales the background mesh by 1.33 per level (cells ~2.4x per level);
# snappy refinement levels are relative to it, so the whole mesh refines consistently.
set -euo pipefail
WORK=$1; NP=${2:-1}
SLANT=${SLANT:-35}; LEVEL=${LEVEL:-medium}; GROUND=${GROUND:-fixed}; ITERS=${ITERS:-1000}
HERE=$(cd "$(dirname "$0")" && pwd)
PSTREAM="$CONDA_PREFIX/lib/mpich-3.3/libPstream.so"   # see cylinder2d/run.sh
par() { if [ "$NP" -gt 1 ]; then mpirun -np "$NP" -genv LD_PRELOAD "$PSTREAM" "$@" -parallel; else "$@"; fi; }
if [ "${RESUME:-0}" != 1 ]; then
  rm -rf "$WORK"; mkdir -p "$WORK"; cp -r "$HERE/case/." "$WORK/"; cd "$WORK"
  case $LEVEL in coarse) B="66 11 14";; medium) B="88 15 18";; fine) B="117 20 24";; *) echo "bad LEVEL"; exit 2;; esac
  sed -i "s/(88 15 18)/($B)/" system/blockMeshDict
  if [ "$GROUND" = fixed ]; then
    sed -i 's/ground   { type fixedValue; value uniform (40 0 0); }/ground   { type noSlip; }/' 0/U
  fi
  foamDictionary system/controlDict -entry endTime -set "$ITERS" > /dev/null
  foamDictionary system/controlDict -entry writeInterval -set 200 > /dev/null
  mkdir -p constant/triSurface
  python "$HERE/geometry.py" "$SLANT" constant/triSurface/ahmed.stl > log.geometry
  blockMesh > log.blockMesh
  surfaceFeatureExtract > log.surfaceFeatureExtract
  snappyHexMesh -overwrite > log.snappyHexMesh
  checkMesh > log.checkMesh || true
  echo "slant=$SLANT level=$LEVEL ground=$GROUND cells=$(grep -m1 'cells:' log.checkMesh | awk '{print $2}')" > case.info
  if [ "$NP" -gt 1 ]; then
    sed -i "s/numberOfSubdomains .*/numberOfSubdomains $NP; method scotch;/" system/decomposeParDict
    decomposePar -force > log.decomposePar
  fi
  par potentialFoam -writephi > log.potentialFoam 2>&1 || true
else
  cd "$WORK"; foamDictionary system/controlDict -entry endTime -set "$ITERS" > /dev/null
fi
par simpleFoam >> log.simpleFoam 2>&1
if [ "$NP" -gt 1 ]; then reconstructPar -latestTime > log.reconstructPar; fi
echo "done: $WORK"
