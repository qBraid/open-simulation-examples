#!/bin/bash
# Wrap the fan-made production cars, and three DrivAerML cars as controls, into the benchmark's dataset root.
# Orientation fixes per car were read off side and top views (wheelbase, overhangs, mirror position vs published specs).
# usage: bash wrap_all.sh   (on the GPU box, after setup_box.sh)
cd /home/jovyan/aero; D=drivaerml; PY=pn/bin/python
run() { $PY -I wrap_mesh.py "$@" >> wrap.log 2>&1 || echo "FAIL $*" >> wrap.log; }
run fanin/drivaer_9001.stl $D 9001 &              # Tesla Model X
run fanin/drivaer_9002.stl $D 9002 --rot x180 &   # Porsche Taycan: was upside down
run fanin/drivaer_9003.stl $D 9003 --rot x180 &   # Hyundai Ioniq 5: was upside down
run fanin/drivaer_9004.stl $D 9004 --rot y180 --r 0.05 &   # BMW i8: upside down and back to front; open underside needs r = 5 cm
run fanin/drivaer_9006.stl $D 9006 --rot y180 &   # BMW M3 E92: upside down and back to front
run fanin/drivaer_9007.stl $D 9007 &              # BMW M3 E30
# controls: the same wrap applied to DrivAerML cars, CFD fields carried over, so the benchmark scores the wrap itself
for c in 202 419 439; do
  ( run $D/run_$c/drivaer_$c.stl $D 9$c --keep-frame --truth $D/run_$c/boundary_$c.vtp
    cp $D/run_$c/force_mom_$c.csv $D/run_9$c/force_mom_9$c.csv; cp $D/run_$c/geo_ref_$c.csv $D/run_9$c/geo_ref_9$c.csv ) &
done
wait; echo WRAPDONE >> wrap.log
