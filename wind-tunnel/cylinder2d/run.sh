#!/usr/bin/env bash
# Run the Re=100 cylinder validation case.
# Usage: run.sh <coarse|fine> <workdir> [nprocs] [half_width] [inlet_dist]
# Env: SCHEME=linearUpwind|linear (convection, default linearUpwind), MAXCO (default 0.8),
#      T1 (end of the spin-up leg, default 185; the recorded leg is always 15 time units)
# Needs the cfd env active (OpenFOAM v2412 + python-gmsh from conda-forge).
set -euo pipefail
LEVEL=$1; WORK=$2; NP=${3:-1}; HW=${4:-10}; XIN=${5:-10}
SCHEME=${SCHEME:-linearUpwind}; MAXCO=${MAXCO:-0.8}; T1=${T1:-185}
HERE=$(cd "$(dirname "$0")" && pwd)
rm -rf "$WORK"; mkdir -p "$WORK"; cp -r "$HERE/case/." "$WORK/"; cd "$WORK"
python "$HERE/mesh.py" "$LEVEL" mesh.msh "$HW" "$XIN" > log.mesh
if [ "$SCHEME" = linear ]; then sed -i "s/div(phi,U)      Gauss linearUpwind grad(U);/div(phi,U)      Gauss linear;/" system/fvSchemes; fi
foamDictionary system/controlDict -entry maxCo -set "$MAXCO" > /dev/null
foamDictionary system/controlDict -entry endTime -set "$T1" > /dev/null
foamDictionary system/controlDict -entry writeInterval -set "$T1" > /dev/null   # leg 1 must write at its end, or leg 2 restarts from 0
gmshToFoam mesh.msh > log.gmshToFoam
foamDictionary constant/polyMesh/boundary -entry entry0/frontAndBack/type -set empty > /dev/null
foamDictionary constant/polyMesh/boundary -entry entry0/cylinder/type -set wall > /dev/null
checkMesh > log.checkMesh
# conda-forge openfoam=2412 links the *dummy* Pstream via RPATH, so "-parallel" dies with
# "The dummy Pstream library cannot be used in parallel mode". LD_LIBRARY_PATH cannot
# override an RPATH; preloading the MPICH build of libPstream does.
PSTREAM="$CONDA_PREFIX/lib/mpich-3.3/libPstream.so"
run() {  # serial or MPI
  if [ "$NP" -gt 1 ]; then mpirun -np "$NP" -genv LD_PRELOAD "$PSTREAM" pimpleFoam -parallel > "$1" 2>&1; else pimpleFoam > "$1" 2>&1; fi
}
if [ "$NP" -gt 1 ]; then sed -i "s/numberOfSubdomains .*/numberOfSubdomains $NP;/" system/decomposeParDict; decomposePar -force > log.decompose; fi
# leg 1: develop the vortex street (t = 0..T1)
run log.pimpleFoam.1
# leg 2: t = T1..T1+15, write fields every 0.25 for the viewer animation
foamDictionary system/controlDict -entry endTime -set $((T1 + 15)) > /dev/null
foamDictionary system/controlDict -entry writeInterval -set 0.25 > /dev/null
run log.pimpleFoam.2
if [ "$NP" -gt 1 ]; then reconstructPar -time "$T1:" > log.reconstruct; fi
echo "done: $WORK"
