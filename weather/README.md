# Global weather: an AI forecast on one GPU, scored against ECMWF

**Open-source stack:** NVIDIA Earth2Studio 0.19 (Apache-2.0) running SFNO, also called FourCastNet v2
(73 channels, 0.25°, 6-hour steps). The weights are commercially usable. ERA5 comes from the public
ARCO zarr store.
**Stands in for:** operational numerical weather prediction (ECMWF IFS HRES) and commercial forecast feeds.
**Viewer:** `viewer.html`, a self-contained 3D globe (open it in a browser or with `qbraid-canvas weather/viewer.html`).

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

**FourCastNet 3 (the HRES-class open model): blocked by GPU memory, not by the pipeline.**
- FCN3 needs CUDA DISCO kernels that the PyPI `torch-harmonics` wheel lacks. We built them from source on the
  pool with pip's CUDA 13.0 toolchain (recipe below); a DISCO test convolution then runs on the GPU.
- The full model still runs out of memory on the 22 GB L4. Its decoder (a transposed DISCO convolution back to
  the 0.25° grid) asks for one 20.4 GB buffer on the first step.
- Next step: rerun `forecast_score.py fcn3` on a `gpu-a100-sxm` (40 GB or more, $2.49/h). The 12-start set should
  take about 20–30 min, roughly $1–1.50. That run decides whether this example reaches the bar.

**Storm case: Hurricane Laura (start 24 Aug 2020).** SFNO's track stays within 170 km of ERA5 for 48 h
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
# FCN3 additionally needs torch-harmonics built with CUDA kernels (the PyPI wheel is CPU-only):
env/bin/uv pip install --python env/bin/python "nvidia-cuda-nvcc==13.0.*" "nvidia-nvvm==13.0.*" \
  "nvidia-cuda-crt==13.0.*" "nvidia-cuda-cccl==13.0.*" ninja setuptools wheel
CU=$(env/bin/python -c "import site;print(site.getsitepackages()[0])")/nvidia/cu13
mkdir -p $CU/lib64 && ln -sf $CU/lib/libcudart.so.13 $CU/lib64/libcudart.so   # the linker wants -lcudart
git clone --depth 1 --branch v0.9.2 https://github.com/NVIDIA/torch-harmonics.git && cd torch-harmonics
CUDA_HOME=$CU PATH=$CU/bin:$PATH FORCE_CUDA_EXTENSION=1 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 \
  ../env/bin/python -m pip install --no-build-isolation --no-deps . && cd ..

python forecast_score.py sfno results/          # 12 starts in 2020, GPU, about 17 min on an L4
python forecast_score.py fcn3 results/          # same starts with FourCastNet 3
python hres_score.py results/                   # IFS HRES on the same starts (CPU, network-bound)
python make_verdict.py results/ results/<model>_viz_2020082400.npz --model <sfno|fcn3>
python build_viewer.py results/ results/<model>_viz_2020082400.npz --model <sfno|fcn3>
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
