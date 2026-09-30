#!/usr/bin/env bash
# Run the Re=100 cylinder validation case.
# Usage: run.sh <coarse|fine> <workdir> [nprocs] [half_width]
# Needs the cfd env active (OpenFOAM v2412 + python-gmsh from conda-forge).
set -euo pipefail
LEVEL=$1; WORK=$2; NP=${3:-1}; HW=${4:-10}
HERE=$(cd "$(dirname "$0")" && pwd)
rm -rf "$WORK"; mkdir -p "$WORK"; cp -r "$HERE/case/." "$WORK/"; cd "$WORK"
python "$HERE/mesh.py" "$LEVEL" mesh.msh "$HW" > log.mesh
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
# leg 1: develop the vortex street (t = 0..185)
run log.pimpleFoam.1
# leg 2: t = 185..200, write fields every 0.25 for the viewer animation
foamDictionary system/controlDict -entry endTime -set 200 > /dev/null
foamDictionary system/controlDict -entry writeInterval -set 0.25 > /dev/null
run log.pimpleFoam.2
if [ "$NP" -gt 1 ]; then reconstructPar -time '185:' > log.reconstruct; fi
echo "done: $WORK"
