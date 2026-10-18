#!/bin/bash
# GPU box setup for the NVIDIA PhysicsNeMo-CFD DrivAerML benchmark, with every fix this run needed.
# Tested on A100 with NVIDIA driver 570 (CUDA 12.8) and 595. usage: bash setup_box.sh   (in /home/jovyan/aero, with val_ids.txt)
set -x
cd /home/jovyan/aero
D=/home/jovyan/aero/drivaerml
PCFD_SHA=0612ec4ed54484a47bfa134eda7b3b012a607624
# 48 validation cars: surface CFD (~660 MB each), STL (DoMINO reads it), forces and reference geometry
( cat val_ids.txt | xargs -P 12 -I{} sh -c 'mkdir -p '$D'/run_{} && cd '$D'/run_{} && for f in boundary_{}.vtp drivaer_{}.stl force_mom_{}.csv geo_ref_{}.csv; do curl -sSfL -C - -o $f https://huggingface.co/datasets/neashton/drivaerml/resolve/main/run_{}/$f || echo FAIL {} $f; done' > dl.log 2>&1; echo DLDONE >> dl.log ) &
export VIRTUAL_ENV=/home/jovyan/aero/pn UV_CACHE_DIR=/tmp/uvc
uv venv -p 3.12 pn > env.log 2>&1
git clone -q https://github.com/NVIDIA/physicsnemo-cfd.git && git -C physicsnemo-cfd checkout -q $PCFD_SHA
uv pip install ./physicsnemo-cfd huggingface_hub trimesh scikit-image rtree >> env.log 2>&1
# torch: physicsnemo needs >= 2.10; uv resolves a cu13 build the 570 driver cannot run. 2.11.0+cu128 is the newest cu128 build.
uv pip install --reinstall-package torch --reinstall-package torchvision "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" --index-url https://download.pytorch.org/whl/cu128 >> env.log 2>&1
# warp: PyPI ships the cu13 build; physicsnemo needs >= 1.14 (1.12 lacks LOG_WARNING). The cu12 wheel is on GitHub releases.
curl -sSfL -o /tmp/warp_lang-1.18.0+cu12-py3-none-manylinux_2_28_x86_64.whl https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0+cu12-py3-none-manylinux_2_28_x86_64.whl
uv pip install --reinstall-package warp-lang /tmp/warp_lang-1.18.0+cu12-py3-none-manylinux_2_28_x86_64.whl >> env.log 2>&1
# cuDNN: a cu13 cuDNN wheel overwrites the cu12 one in the same directory (CUDNN_STATUS_NOT_INITIALIZED)
uv pip uninstall nvidia-cudnn-cu13 >> env.log 2>&1; uv pip install --reinstall-package nvidia-cudnn-cu12 "nvidia-cudnn-cu12==9.19.0.56" >> env.log 2>&1
uv pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.11.0+cu128.html >> env.log 2>&1
# KNN: without cuML, physicsnemo falls back to brute-force torch KNN, O(N*M) memory (DoMINO asks for ~740k GiB). Use scipy's KD-tree.
K=pn/lib/python3.12/site-packages/physicsnemo/nn/functional/neighbors/knn/knn.py
python3 - $K <<'PY'
import sys; f=sys.argv[1]; s=open(f).read()
old='''        if impl is None:
            # Get the torch implementation
            impl = impls["torch"]'''
new='''        if impl is None and points.is_cuda and impls["scipy"].available:
            # qBraid shim: no cuML here; brute-force torch KNN needs O(N*M) memory. KD-tree on CPU instead.
            i_, d_ = impls["scipy"].func(points.cpu(), queries.cpu(), k)
            return i_.to(points.device), d_.to(points.device)
        if impl is None:
            # Get the torch implementation
            impl = impls["torch"]'''
assert old in s; open(f,"w").write(s.replace(old,new)); print("knn patched")
PY
# ... and the scipy path queries on one thread by default: use every core (128 on these boxes); GeoTransolver 6.5 -> ? min per car
sed -i "s/distance, indices = interp_func.query(queries, k=k)$/distance, indices = interp_func.query(queries, k=k, workers=-1)/" ${K%knn.py}_scipy_impl.py
grep -q "workers=-1" ${K%knn.py}_scipy_impl.py && echo "knn threaded"
pn/bin/python -c "import torch,warp as wp;wp.init();print(torch.__version__,torch.cuda.is_available(),torch.backends.cudnn.version(),wp.__version__);import torch_scatter,physicsnemo.cfd;print('imports ok')" >> env.log 2>&1
uv pip freeze > freeze.txt
echo ENVDONE >> env.log
wait
