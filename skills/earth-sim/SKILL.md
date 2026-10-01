---
name: earth-sim
description: Run and score AI global weather forecasts (NVIDIA Earth2Studio with SFNO / FourCastNet) on qBraid GPUs from ERA5 initial conditions, benchmark against WeatherBench2 and ECMWF IFS HRES, and render an interactive globe. Use for medium-range forecasting, forecast verification, weather/climate demos, or comparing AI weather models.
---

# Earth system simulation on qBraid (Earth2Studio)

## What runs where
| Task | Machine | Notes |
|---|---|---|
| SFNO 73-ch, 0.25°, 5-day forecast | `gpu-l4` (22 GB usable) is enough for SFNO | ~1.2 s per 6 h step on an L4 after warm-up; first model load ~100 s (NGC download) |
| FCN3 (needs >22 GB), ensembles | `gpu-a100-sxm` / `gpu-h100-sxm` | ensembles scale linearly in memory; start with 4 members on an L4 |
| Scoring, plotting, viewer build | CPU or the Lab pod | xarray + numpy only |

## Install (verified 2026-10-01)
Earth2Studio 0.19 does **not** install cleanly with a bare `pip install earth2studio[sfno]`:
1. `makani` is not on PyPI. Install it from the pin in Earth2Studio's own pyproject:
   `"makani @ git+https://github.com/NVIDIA/makani.git@b38fcb2799d7dbc146fa60459f3f9823394a8bf1"`.
2. The resolver picks the newest torch (2.14), but `torch-harmonics` wheels are built per torch minor
   (0.9.2 needs torch 2.11) and fail with `undefined symbol ... torch7Library4_def`. Pin
   `torch==2.11.* torch-harmonics==0.9.2 torchvision==0.26.*` (a mismatched torchvision fails with
   `operator torchvision::nms does not exist`).
3. The env is ~12 GB. On GPU boxes whose home is a network filesystem, put it on local disk if there is
   room. Otherwise expect a slow, one-off install. Delete the uv/pip cache afterwards.
4. Set `EARTH2STUDIO_CACHE` to a large disk for the weights, but use `ARCO(cache=False)` for ERA5: an ARCO
   chunk carries every pressure level, so caching costs about 11 GB per start date.
5. **FCN3 needs CUDA DISCO kernels.** The PyPI `torch-harmonics` wheel is CPU-only (`+torch2.11.0.cpu`), and
   FCN3 fails with `Could not run 'disco_kernels::forward' with arguments from the 'CUDA' backend`. SFNO is
   unaffected. To fix it, build from the GitHub tag with `FORCE_CUDA_EXTENSION=1 --no-build-isolation`, using the
   pip CUDA toolchain (`nvidia-cuda-nvcc`, `nvidia-cuda-cccl`, `CUDA_HOME=<site-packages>/nvidia/cu13`). Pin
   `nvidia-nvvm` and `nvidia-cuda-crt` to the **same** minor as `nvidia-cuda-nvcc`. torch 2.14 pulls nvvm 13.4,
   and the build then dies with `ptxas fatal: Unsupported .version 9.4; current version is '9.0'`.

## Licences: only commercially usable weights
- OK: SFNO / FourCastNet (NVIDIA), FCN3, Aurora (MIT).
- Not OK for commercial demos: GraphCast weights (CC BY-NC-SA), Pangu weights (non-commercial).
  Their *published scores* may be cited from WeatherBench2.

## Benchmark discipline (mandatory)
- Truth = ERA5 (ARCO zarr, `earth2studio.data.ARCO`). Metric = latitude-weighted global RMSE at 0.25°,
  z500 / T850 / T2m at day 1, 3 and 5, which is the WeatherBench2 headline set.
- Score **IFS HRES on the same init dates** from WeatherBench2's public zarr
  (`gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr`, anonymous). A paired comparison removes
  the sampling noise of a small init set. Also cite the WB2 full-year numbers
  (`gs://weatherbench2/benchmark_results/*_vs_era5_1440x721_2020.nc`).
- Evaluate on years outside the model's training period (2018+ for SFNO). WB2's year is 2020.
- Note that HRES scored against ERA5 is penalised at short leads (different analysis). WB2 does the same.

## Recipe
`weather/forecast_score.py <sfno|fcn3> <out>` → `weather/hres_score.py <out>` → `weather/build_viewer.py <out> <viz.npz>`.
Run the GPU step through the queue on shared boxes.

## Choosing the model (the decision that matters)
- **`SFNO.load_default_package()` is `sfno_73ch_small`**, NVIDIA's public small checkpoint (embed 384, 8 layers).
  It is a good demo model, but it is not HRES-class beyond day 1 (see below). Say so before a user quotes skill numbers.
- For skill, use FCN3 (FourCastNet 3). It needs the torch-harmonics CUDA build above **and a GPU with 40 GB or more**:
  on a 22 GB L4 its decoder OOMs on the first step (one 20.4 GB allocation). Use `gpu-a100-sxm` or larger.
- Always state which checkpoint ran. "SFNO" alone is ambiguous.

## Verified result (2026-10-01, gpu-l4, 12 starts in 2020, 5-day leads)
| | z500 day 1/3/5 (m²/s²) | T850 day 5 (K) | T2m day 5 (K) |
|---|---|---|---|
| sfno_73ch_small (this run) | 66 / 208 / 407 | 2.39 | 2.01 |
| IFS HRES, same starts (our scorer) | 49 / 134 / 299 | 1.93 | 1.73 |
| IFS HRES, WB2 2020 full year | 49 / 139 / 308 | 1.94 | 1.73 |

- Our scorer reproduces WB2's HRES numbers, so the pipeline is right. SFNO-small is 36% worse than HRES on
  day-5 z500 and only better on day-1 near-surface temperature.
- The 12-start run took 17 GPU-minutes (~$0.15 of L4). The ERA5 fetch, not the GPU, dominates wall time.

## Visuals
`weather/viewer.html`: a three.js globe with an atmospheric glow and day/night terminator, wind particles
advected from the forecast 850 hPa winds, z500 contours, a lead-time scrubber, a forecast vs ERA5 swipe,
an error map, storm tracks, and a guided tour.
- Fields are 1°, winds 2°, 8-bit with row deltas, all in one deflate stream: 5.0 MB.
- The Agent Canvas rejects about 5 MB (HTTP 413), so keep payloads under that.
- Verify inside a `sandbox="allow-scripts"` srcdoc frame with `weather/verify_viewer.py` (Playwright waits on
  `window.__ready` and freezes the render loop via `window.__freeze` before each screenshot).
