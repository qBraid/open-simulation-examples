#!/bin/bash
# End-to-end: environment, AC-OPF benchmark, expansion scenarios, viewer.
# On the shared pool box, wrap heavy steps in /tmp/ose/cpu-run (see /tmp/ose/README).
set -euo pipefail
ENV=${ENV:-/tmp/ose/energy-grid/env}
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-/tmp/ose/mamba} PATH=$ENV/bin:$PATH OMP_NUM_THREADS=1
HERE=$(cd "$(dirname "$0")" && pwd); WORK=${WORK:-/tmp/ose/energy-grid}; mkdir -p "$WORK"; cd "$WORK"

[ -x "$ENV/bin/python" ] || { micromamba create -y -p "$ENV" -c conda-forge python=3.12 ipopt=3.14 pyomo pypsa highspy networkx scipy pandas numpy netcdf4
                            "$ENV/bin/pip" install gridx-egret "pandapower>=3"; }
[ -d pglib-opf ] || git clone -q --depth 1 https://github.com/power-grid-lib/pglib-opf.git
[ -f scigrid_de.nc ] || python -c "import pypsa; pypsa.examples.scigrid_de().export_to_netcdf('scigrid_de.nc')"

python "$HERE/opf/acopf_pglib.py" pglib-opf out \
  case14_ieee case57_ieee case118_ieee case300_ieee case500_goc case1354_pegase case1888_rte case2000_goc \
  case2869_pegase case3012wp_k case118_ieee__api case1354_pegase__api case2869_pegase__api \
  case118_ieee__sad case1354_pegase__sad case2869_pegase__sad

for s in w1.0_s1.0_c0 w1.0_s1.0_c100 w1.5_s1.0_c0 w2.0_s1.0_c100; do
  python "$HERE/expansion/expansion.py" scigrid_de.nc exp "$s" --pf
done

mkdir -p "$HERE/results/opf" "$HERE/results/expansion"
cp out/*.json "$HERE/results/opf/"; cp exp/*.json "$HERE/results/expansion/"
cd "$HERE" && python build_viewer.py && python make_report.py
