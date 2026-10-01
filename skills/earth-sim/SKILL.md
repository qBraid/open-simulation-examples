---
name: earth-sim
description: Run and score AI global weather forecasts (NVIDIA Earth2Studio with SFNO / FourCastNet) on qBraid GPUs from ERA5 initial conditions, benchmark against WeatherBench2 and ECMWF IFS HRES, and render an interactive globe. Use for medium-range forecasting, forecast verification, weather/climate demos, or comparing AI weather models.
---

# Earth system simulation on qBraid (Earth2Studio)

## What runs where
| Task | Machine | Notes |
|---|---|---|
| SFNO 73-ch, 0.25°, 5-day forecast | `gpu-l4` (24 GB) is enough | ~__STEP__ s per 6 h step on an L4; model load ~__LOAD__ s |
| FCN3, larger models, ensembles | `gpu-a100-sxm` / `gpu-h100-sxm` | ensembles scale linearly in memory; start with 4 members on an L4 |
| Scoring, plotting, viewer build | CPU or the Lab pod | xarray + numpy only |

## Install (verified __DATE__)
Earth2Studio 0.19 does **not** install cleanly with a bare `pip install earth2studio[sfno]`:
1. `makani` is not on PyPI. Install it from the pin in Earth2Studio's own pyproject:
   `"makani @ git+https://github.com/NVIDIA/makani.git@b38fcb2799d7dbc146fa60459f3f9823394a8bf1"`.
2. The resolver picks the newest torch (2.14), but `torch-harmonics` wheels are built per torch minor
   (0.9.2 needs torch 2.11) and fail with `undefined symbol ... torch7Library4_def`. Pin
   `torch==2.11.* torch-harmonics==0.9.2 torchvision==0.26.*` (a mismatched torchvision fails with
   `operator torchvision::nms does not exist`).
3. The env is ~12 GB. On GPU boxes whose home is a network filesystem, put it on local disk if there is
   room. Otherwise expect a slow, one-off install. Delete the uv/pip cache afterwards.
4. Set `EARTH2STUDIO_CACHE` to a large disk. Weights and ERA5 chunks are cached there.

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

## Verified result (__DATE__)
__RESULT__
