# Global weather: an AI forecast on one GPU, scored against ECMWF

**Open-source stack:** NVIDIA Earth2Studio 0.19 (Apache-2.0) running SFNO, also called FourCastNet v2
(73 channels, 0.25°, 6-hour steps). The weights are commercially usable. ERA5 comes from the public
ARCO zarr store.
**Stands in for:** operational numerical weather prediction (ECMWF IFS HRES) and commercial forecast feeds.
**Viewer:** `viewer.html` (5.0 MB), a self-contained 3D globe with an FCN3 / SFNO-small model toggle (open it in a browser or with `qbraid-canvas weather/viewer.html`).

## What "top-10%" means here
The public yardstick is [WeatherBench2](https://sites.research.google/weatherbench/) (WB2). The metric is
latitude-weighted global RMSE against ERA5 at 0.25° for z500, T850 and T2m at day 1, 3 and 5, in the
2020 test year. On WB2 the leaders are GraphCast and recent successors, then Pangu, then **IFS HRES**,
ECMWF's operational deterministic model. Matching or beating IFS HRES puts a forecast in the top tier
of global deterministic systems.

So the bar here is: **at day 3 to 5, SFNO run on qBraid is at least as good as IFS HRES on the same start
dates, and in line with its published skill.** SFNO is not on WB2, so its "published skill" comes from
the SFNO paper. We also show where the WB2 leaders sit.

## Results (verified 2026-10-01, qBraid `gpu-l4`)

**Set-up.** 12 forecasts, one 00 UTC start in each month of 2020 (WB2's test year, outside SFNO's
training). Each run is 5 days in 20 autoregressive 6 h steps on one NVIDIA L4. Truth is ERA5. Scores
are latitude-weighted global RMSE at 0.25°.

**Scoring check.** IFS HRES scored by our pipeline on these 12 dates gives 49 / 134 / 299 m²/s² for z500
at day 1/3/5. WB2's published full-year 2020 numbers are 49 / 139 / 308. The scorer reproduces the
reference, so the comparison below is apples to apples.

| Variable | Lead | SFNO (12 starts) | IFS HRES (same starts) | Paired diff ± se | SFNO better | WB2 HRES 2020 | WB2 GraphCast 2020 |
|---|---|---|---|---|---|---|---|
| z500 (m²/s²) | day 1 | 66 | 49 | +17.3 ± 1.9 | 0/12 | 49 | 41 |
| z500 | day 3 | 208 | 134 | +73.7 ± 4.4 | 0/12 | 139 | 126 |
| z500 | day 5 | 407 | 299 | +108 ± 17 | 1/12 | 308 | 276 |
| T850 (K) | day 1 | 0.82 | 0.85 | −0.02 ± 0.01 | 8/12 | 0.86 | 0.60 |
| T850 | day 3 | 1.53 | 1.28 | +0.25 ± 0.02 | 0/12 | 1.30 | 1.03 |
| T850 | day 5 | 2.39 | 1.93 | +0.46 ± 0.07 | 0/12 | 1.94 | 1.63 |
| T2m (K) | day 1 | 0.89 | 1.12 | −0.23 ± 0.02 | 12/12 | 1.14 | 0.63 |
| T2m | day 3 | 1.43 | 1.35 | +0.08 ± 0.03 | 2/12 | 1.37 | 0.94 |
| T2m | day 5 | 2.01 | 1.73 | +0.28 ± 0.05 | 1/12 | 1.73 | 1.36 |

**Verdict for SFNO: not reached.** Earth2Studio's default SFNO is NVIDIA's public
`sfno_73ch_small` checkpoint (embed 384, 8 layers), not the large model from the SFNO paper. It beats
HRES only on near-surface temperature at day 1. By day 5 it is 36% worse on z500 and 24% worse on T850.
It is a good demonstration model, not a top-tier forecaster.

### FourCastNet 3: the verdict (verified 2026-10-01, qBraid `gpu-a100-sxm`, A100 80GB)

FCN3 is a **probabilistic** model: each run draws a different but plausible future. Two fair comparisons follow.

**One member against deterministic HRES (12 starts).** z500 at day 1/3/5 is 59 / 180 / 394 m²/s² against HRES 49 / 134 / 299.
A single FCN3 member is not meant to beat a deterministic forecast on RMSE. It matches **an IFS ensemble member**
(WB2 2020: 65 / 197 / 397) and it beats SFNO-small at every lead.

**4-member ensemble mean against HRES, paired on 6 of the starts (Jan, Mar, May, Jul, Sep, Nov):**

| Variable | Lead | FCN3 4-member mean | IFS HRES (same starts) | Paired diff ± se | FCN3 better | WB2 IFS ENS mean (50 members) |
|---|---|---|---|---|---|---|
| z500 (m²/s²) | day 1 | 46.2 | 49.6 | −3.5 ± 1.3 | 5/6 | 46.8 |
| z500 | day 3 | 139.7 | 133.3 | **+6.4 ± 2.2** | 1/6 | 134.8 |
| z500 | day 5 | 298.5 | 296.3 | +2.2 ± 12.0 | 2/6 | 280.2 |
| T850 (K) | day 1 | 0.70 | 0.85 | −0.15 ± 0.01 | 6/6 | 0.81 |
| T850 | day 3 | 1.17 | 1.29 | −0.12 ± 0.02 | 6/6 | 1.18 |
| T850 | day 5 | 1.74 | 1.91 | −0.17 ± 0.04 | 6/6 | 1.69 |
| T2m (K) | day 1 | 0.79 | 1.13 | −0.34 ± 0.03 | 6/6 | 0.98 |
| T2m | day 3 | 1.10 | 1.34 | −0.24 ± 0.05 | 6/6 | 1.20 |
| T2m | day 5 | 1.47 | 1.69 | −0.23 ± 0.01 | 6/6 | 1.50 |

**Verdict: close.** On 7 of 9 variable–lead pairs the FCN3 ensemble mean is better than ECMWF's operational HRES.
- Temperature at 850 hPa and 2 m is better at every lead on every start.
- Day-5 z500 is level within noise. The one clear miss is **day-3 z500, 5% worse** (+6.4 ± 2.2).
- So the bar ("as good as HRES at day 3–5") is met everywhere except mid-range z500.

Caveats:
- An ensemble mean is smoother than a single forecast, which helps RMSE. The like-for-like ensemble yardstick is
  ECMWF's 50-member ensemble mean, and 4 FCN3 members are close to it on temperature but behind on day-5 z500
  (298 vs 280).
- More members would close part of that gap. With 12 starts × 8 members the z500 error bars would also tighten.

**Hurricane Laura with FCN3 (one member).** The track error is 38 km at 24 h, 56 km at 48 h and 185 km at 72 h, and
the central pressure stays within 4–5 hPa. FCN3 turns Laura north toward Louisiana as the real storm did, where
SFNO-small kept it heading west (462 km at 72 h, 813 km at 84 h).

**Cost.**
- About 2.6 s per 6 h step on the A100; 54 s model load.
- 12 single-member starts took 21 min and the 4-member ensemble on 6 starts took 25 min. The SFNO viewer rerun reproduced the
  L4 scores to the last digit.
- About 1.6 h of A100 time in total including the build, ≈ $4. The `torch-harmonics` CUDA build is a one-off (recipe below).

**Storm case: Hurricane Laura (start 24 Aug 2020), SFNO-small.** SFNO's track stays within 170 km of ERA5 for 48 h
and 462 km at 72 h, with central pressure within 2–3 hPa. It then keeps Laura moving west across the
Gulf while the real storm recurves north into Louisiana: the error is 813 km at 84 h. The viewer's
tour shows this.

**Speed and cost.** About 1.2 s per 6 h global step on an L4 after warm-up, and about 100 s to load the
model the first time. The 12-start SFNO run took 17 minutes, roughly $0.15 of L4 time. Fetching ERA5
from the public cloud store dominates the wall time, not the GPU.


## Reproduce
```bash
# environment (see skills/earth-sim for the pins and why they are needed)
python -m venv env && env/bin/pip install uv
env/bin/uv pip install --python env/bin/python "earth2studio[sfno]==0.19.0" \
  "makani @ git+https://github.com/NVIDIA/makani.git@b38fcb2799d7dbc146fa60459f3f9823394a8bf1" \
  "torch==2.11.*" "torch-harmonics==0.9.2" "torchvision==0.26.*" scipy
export EARTH2STUDIO_CACHE=/big/disk/e2s-cache
# FCN3 additionally needs torch-harmonics built with CUDA kernels (the PyPI wheel is CPU-only).
# Match the toolkit to the driver: driver >= 580 -> CUDA 13 (pip nvcc, below); driver 570 (e.g. qBraid A100) -> CUDA 12.8:
#   torch==2.11.* from https://download.pytorch.org/whl/cu128, and nvcc from conda-forge (the pip cu12 nvcc wheel has no nvcc):
#   micromamba create -p /tmp/cuda -c conda-forge cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-cccl=12.8 cuda-version=12.8 \
#       libcusparse-dev libcublas-dev libcusolver-dev libcurand-dev cuda-nvtx-dev cuda-profiler-api
#   CUDA_HOME=/tmp/cuda PATH=/tmp/cuda/bin:$PATH LIBRARY_PATH=/tmp/cuda/lib FORCE_CUDA_EXTENSION=1 TORCH_CUDA_ARCH_LIST=8.0 \
#       pip install --no-build-isolation --no-deps ./torch-harmonics
# CUDA 13 variant:
env/bin/uv pip install --python env/bin/python "nvidia-cuda-nvcc==13.0.*" "nvidia-nvvm==13.0.*" \
  "nvidia-cuda-crt==13.0.*" "nvidia-cuda-cccl==13.0.*" ninja setuptools wheel
CU=$(env/bin/python -c "import site;print(site.getsitepackages()[0])")/nvidia/cu13
mkdir -p $CU/lib64 && ln -sf $CU/lib/libcudart.so.13 $CU/lib64/libcudart.so   # the linker wants -lcudart
git clone --depth 1 --branch v0.9.2 https://github.com/NVIDIA/torch-harmonics.git && cd torch-harmonics
CUDA_HOME=$CU PATH=$CU/bin:$PATH FORCE_CUDA_EXTENSION=1 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 \
  ../env/bin/python -m pip install --no-build-isolation --no-deps . && cd ..

python forecast_score.py sfno results/          # 12 starts in 2020, GPU, about 17 min on an L4
python forecast_score.py fcn3 results/          # same starts with FourCastNet 3 (needs a GPU with >22 GB)
python ensemble_score.py results/ --members 4   # FCN3 ensemble mean on 6 starts
python hres_score.py results/                   # IFS HRES on the same starts (CPU, network-bound)
python make_verdict.py results/ results/fcn3_viz_2020082400.npz --model fcn3
python build_viewer.py results/ --models fcn3=results/fcn3_viz_2020082400.npz,sfno=results/sfno_viz_2020082400.npz
python verify_viewer.py viewer.html results/   # sandboxed-frame check + screenshots (Playwright)
```
The WB2 reference numbers in `results/wb2_ref_2020.json` come from
`gs://weatherbench2/benchmark_results/*_vs_era5_1440x721_2020.nc`.

## Honest limits
- **Twelve start dates, not 730.** The paired comparison against HRES on exactly those dates removes most
  of the date-to-date noise, but the absolute RMSEs carry the sampling spread shown as bars in the viewer.
- **HRES against ERA5** is slightly penalised at day 1, because its own analysis is not ERA5. WB2 has the same caveat.
- **Smoothing.** Like other ML models trained on mean-squared error, SFNO blurs small scales as lead time
  grows. That lowers RMSE but understates extremes: hurricane central pressure is too weak.
- **Initial conditions.** ERA5 initial conditions are a reanalysis available days later. A real-time
  service would start from GFS or IFS analyses, so live skill is somewhat lower.
- **Licences.** GraphCast and Pangu are shown only as published WB2 scores. Their weights are non-commercial
  and are not used here.
