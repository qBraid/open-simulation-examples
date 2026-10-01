---
name: earth-sim
description: Run and score AI global weather forecasts (NVIDIA Earth2Studio with SFNO / FourCastNet) on qBraid GPUs from ERA5 initial conditions, benchmark against WeatherBench2 and ECMWF IFS HRES, and render an interactive globe. Use for medium-range forecasting, forecast verification, weather/climate demos, or comparing AI weather models.
---

# Earth system simulation on qBraid (Earth2Studio)

## What runs where
| Task | Machine | Notes |
|---|---|---|
| SFNO 73-ch, 0.25°, 5-day forecast | `gpu-l4` (22 GB usable) is enough for SFNO | ~1.2 s per 6 h step on an L4 after warm-up; first model load ~100 s (NGC download) |
| FCN3 (needs >22 GB; ~70 GB peak), ensembles | `gpu-a100-sxm` (2.6 s/step) / `gpu-h100-sxm` | ensembles scale linearly in memory; start with 4 members on an L4 |
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
   unaffected. To fix it on CUDA 13 hosts (driver ≥580; see below for driver 570), build from the GitHub tag with `FORCE_CUDA_EXTENSION=1 --no-build-isolation`, using the
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
  It is a good demo model, but it is not HRES-class beyond day 1. Say so before a user quotes skill numbers.
- **FCN3 (FourCastNet 3) is the skill model.** It is *probabilistic*, so score it the right way:
  - a **single member** compares with an IFS *ensemble member*, not with deterministic HRES;
  - an **ensemble mean** (`set_rng(seed=…)` per member) compares with HRES and with the IFS ENS mean.
- **FCN3 needs >22 GB of GPU memory.** On the L4 its decoder asks for a 20.4 GB block on the first step and OOMs.
  Use `gpu-a100-sxm`; it peaks around 70 GB at 0.25° and runs at 2.6 s per 6 h step.
- **The torch-harmonics CUDA build must match the host driver.** Driver 570 means CUDA 12.8: torch cu128 wheels plus
  conda-forge `cuda-nvcc=12.8` and the CUDA math dev headers. The pip cu12 nvcc wheel contains only ptxas.
  Driver ≥580 means the CUDA 13 pip toolchain. Check with
  `nvidia-smi --query-gpu=driver_version --format=csv,noheader` first.
- **Earth2Studio's `create_iterator` modifies its input tensor in place.** Pass `x.clone()` for every ensemble
  member. Without it, members after the first start from a corrupted state and score at climatology level (z500 ≈ 1100 at day 1).
- Always state which checkpoint ran. "SFNO" alone is ambiguous.

## Verified results (2026-10-01; 12 starts in 2020, paired with IFS HRES on the same starts)
| | z500 day 1/3/5 (m²/s²) | T850 day 5 (K) | T2m day 5 (K) | Machine |
|---|---|---|---|---|
| sfno_73ch_small, 1 run | 66 / 208 / 407 | 2.39 | 2.01 | L4 |
| FCN3, 1 member | 59 / 180 / 394 | 2.27 | 1.91 | A100 |
| **FCN3, 4-member mean** (6 starts) | **46 / 140 / 299** | **1.74** | **1.47** | A100 |
| IFS HRES, same starts | 49 / 134 / 299 (6 starts: 50 / 133 / 296) | 1.93 | 1.73 | WB2 zarr |
| IFS ENS member / 50-member mean, WB2 2020 | 65 / 197 / 397 · 47 / 135 / 280 | 2.34 · 1.69 | 1.93 · 1.50 | WB2 |

- Our scorer reproduces WB2's HRES numbers, so the pipeline is right.
- The FCN3 4-member mean beats HRES on 7 of 9 variable–lead pairs: all temperature pairs and day-1 z500. Day-5 z500
  is level; **day-3 z500 is 5% worse**. Verdict: *close* to HRES-class with 4 members.
- Hurricane Laura with FCN3: track error 185 km at 72 h, and it turns north correctly. SFNO-small does not (462 km).

## Visuals
`weather/viewer.html`: a three.js globe with an atmospheric glow and day/night terminator, wind particles
advected from the forecast 850 hPa winds, z500 contours, a lead-time scrubber, a forecast vs ERA5 swipe,
an error map, storm tracks, and a guided tour.
- Fields are 1°, winds 2°, 8-bit with row deltas, all in one deflate stream: 5.0 MB.
- The Agent Canvas rejects about 5 MB (HTTP 413), so keep payloads under that.
- Verify inside a `sandbox="allow-scripts"` srcdoc frame with `weather/verify_viewer.py` (Playwright waits on
  `window.__ready` and freezes the render loop via `window.__freeze` before each screenshot).
