---
name: earth-sim
description: Run and score AI global weather forecasts (NVIDIA Earth2Studio with SFNO / FourCastNet 3) on qBraid GPUs from ERA5 initial conditions, benchmark against WeatherBench2 and ECMWF IFS HRES, and render an interactive globe. Use for medium-range forecasting, forecast verification, weather or climate demos, or comparing AI weather models with operational NWP.
metadata:
  version: "0.3.0"
  status: "provisional"
  verified: "2026-10-01"
---

# Earth system simulation on qBraid (Earth2Studio)

AI global weather forecasts from ERA5 initial conditions, scored the same way
WeatherBench2 scores operational models.

Start from the user's goal: if it is unclear whether they want the best
solution or the best quantum solution, ask (see solution-router).

## Get the scripts

The worked example lives in `weather/` of
https://github.com/qBraid/open-simulation-examples:

```bash
git clone --depth 1 https://github.com/qBraid/open-simulation-examples
```

All script paths below are relative to that repo. Its README has the full
reproduce block and every result.

## What runs where

Pick a profile from `qbraid compute list`, launch it through the
qbraid-cloud-orchestration skill with `--auto-stop`, and terminate it when
done.

| Task | Instance | Notes |
|---|---|---|
| SFNO 73-ch, 0.25°, 5-day forecast | `gpu-l4` (22 GB usable) | ~1.2 s per 6 h step after warm-up; first model load ~100 s (NGC download) |
| FCN3 (needs >22 GB; ~70 GB peak), ensembles | `gpu-a100-sxm` (2.6 s/step) or `gpu-h100-sxm` | Ensembles scale linearly in memory |
| Scoring, plotting, viewer build | CPU or the Lab pod | xarray + numpy only |

## Environment (verified 2026-10-01)

No qBraid environment carries this stack yet. Check `qbraid envs available`
first. Otherwise build it as below (the exact commands are in
`weather/README.md`), and package it with the manage-environments skill for
reuse. Earth2Studio 0.19 does **not** install cleanly with a bare
`pip install earth2studio[sfno]`:

1. `makani` is not on PyPI. Install it from the pin in Earth2Studio's own pyproject:
   `"makani @ git+https://github.com/NVIDIA/makani.git@b38fcb2799d7dbc146fa60459f3f9823394a8bf1"`.
2. The resolver picks the newest torch (2.14), but `torch-harmonics` wheels are built per torch minor
   (0.9.2 needs torch 2.11) and fail with `undefined symbol ... torch7Library4_def`. Pin
   `torch==2.11.* torch-harmonics==0.9.2 torchvision==0.26.*` (a mismatched torchvision fails with
   `operator torchvision::nms does not exist`).
3. The env is ~12 GB. On GPU instances whose home is a network filesystem, put it on local disk if there is
   room. Otherwise expect a slow, one-off install. Delete the uv/pip cache afterwards.
4. Set `EARTH2STUDIO_CACHE` to a large disk for the weights, but use `ARCO(cache=False)` for ERA5: an ARCO
   chunk carries every pressure level, so caching costs about 11 GB per start date.
5. **FCN3 needs CUDA DISCO kernels.** The PyPI `torch-harmonics` wheel is CPU-only (`+torch2.11.0.cpu`), and
   FCN3 fails with `Could not run 'disco_kernels::forward' with arguments from the 'CUDA' backend`. SFNO is
   unaffected. Build torch-harmonics from its GitHub tag with `FORCE_CUDA_EXTENSION=1 --no-build-isolation`,
   matching the toolkit to the host driver (check with
   `nvidia-smi --query-gpu=driver_version --format=csv,noheader`):
   - **Driver ≥580: CUDA 13 pip toolchain** (`nvidia-cuda-nvcc`, `nvidia-cuda-cccl`,
     `CUDA_HOME=<site-packages>/nvidia/cu13`). Pin `nvidia-nvvm` and `nvidia-cuda-crt` to the **same** minor as
     `nvidia-cuda-nvcc`: torch 2.14 pulls nvvm 13.4, and the build then dies with
     `ptxas fatal: Unsupported .version 9.4; current version is '9.0'`.
   - **Driver 570 (seen on qBraid A100s): CUDA 12.8.** torch cu128 wheels plus conda-forge `cuda-nvcc=12.8`
     and the CUDA math dev headers. The pip cu12 nvcc wheel contains only ptxas.

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
- HRES scored against ERA5 is penalised at short leads (different analysis). WB2 does the same.

## Recipe

`weather/forecast_score.py <sfno|fcn3> <out>` → `weather/ensemble_score.py <out> --members 4` →
`weather/hres_score.py <out>` → `weather/make_verdict.py` → `weather/build_viewer.py`.
GPUs can disappear mid-run on cloud instances, so write per-start results as you go and copy them back
before terminating.

## Choosing the model (the decision that matters)

- **`SFNO.load_default_package()` is `sfno_73ch_small`**, NVIDIA's public small checkpoint (embed 384, 8 layers).
  It is a good demo model, but it is not HRES-class beyond day 1. Say so before a user quotes skill numbers.
- **FCN3 (FourCastNet 3) is the skill model.** It is *probabilistic*, so score it the right way:
  - a **single member** compares with an IFS *ensemble member*, not with deterministic HRES;
  - an **ensemble mean** (`set_rng(seed=…)` per member) compares with HRES and with the IFS ENS mean.
- **FCN3 needs >22 GB of GPU memory.** On the L4 its decoder asks for a 20.4 GB block on the first step and OOMs.
  Use `gpu-a100-sxm`; it peaks around 70 GB at 0.25° and runs at 2.6 s per 6 h step.
- **Earth2Studio's `create_iterator` modifies its input tensor in place.** Pass `x.clone()` for every ensemble
  member. Without it, members after the first start from a corrupted state and score at climatology level (z500 ≈ 1100 at day 1).
- Always state which checkpoint ran. "SFNO" alone is ambiguous.

## Verified results (12 starts in 2020, paired with IFS HRES on the same starts)

| | z500 day 1/3/5 (m²/s²) | T850 day 5 (K) | T2m day 5 (K) | Instance |
|---|---|---|---|---|
| sfno_73ch_small, 1 run | 66 / 208 / 407 | 2.39 | 2.01 | gpu-l4 |
| FCN3, 1 member | 59 / 180 / 394 | 2.27 | 1.91 | gpu-a100-sxm |
| **FCN3, 4-member mean** (6 starts) | **46 / 140 / 299** | **1.74** | **1.47** | gpu-a100-sxm |
| IFS HRES, same starts | 49 / 134 / 299 (6 starts: 50 / 133 / 296) | 1.93 | 1.73 | WB2 zarr |

- The scorer reproduces WB2's HRES numbers, so the pipeline is right.
- The FCN3 4-member mean beats HRES on 7 of 9 variable–lead pairs: all temperature pairs and day-1 z500. Day-5 z500
  is level; **day-3 z500 is 5% worse**. Verdict: *close* to HRES-class with 4 members.
- Hurricane Laura with FCN3: track error 185 km at 72 h, and it turns north correctly. SFNO-small does not (462 km).

## Visuals

`weather/viewer.html`: a three.js globe with an atmospheric glow and day/night terminator, wind particles
advected from the forecast 850 hPa winds, z500 contours, a lead-time scrubber, a forecast vs ERA5 swipe,
an error map, storm tracks, and a guided tour. Render it with the agent-canvas skill.
- Fields are 1°, winds 2°, 8-bit with row deltas, all in one deflate stream: 5.0 MB.
- The Agent Canvas rejected a page of about 5 MB with HTTP 413, so keep payloads under that.
- Verify inside a `sandbox="allow-scripts"` srcdoc frame with `weather/verify_viewer.py` (Playwright waits on
  `window.__ready` and freezes the render loop via `window.__freeze` before each screenshot).

## Quantum: the short answer

**No credible near-term path.** Operational and AI weather models integrate
or emulate PDEs over hundreds of millions of grid values; quantum PDE solvers
would need fault-tolerant hardware and still pay to load and read that state,
and quantum machine learning for weather is research at toy scale. The best
answer today is a GPU AI forecast scored against ERA5 and HRES. Any QPU
experiment for learning purposes is priced with `qbraid devices get` and
confirmed with the user first. For the full answer and the experiment
protocol, use the **quantum-readiness** skill (sections "Machine learning and
control" and "Differential equations and simulation").

## Verification stamp

- 2026-10-01: Earth2Studio 0.19.0, torch 2.11, torch-harmonics 0.9.2 (CUDA build for FCN3), makani at the pinned commit.
- SFNO: `gpu-l4`, 12 starts in 17 min, roughly $0.15.
- FCN3: `gpu-a100-sxm` (driver 570), 12 single-member starts in 21 min, 4-member ensemble on 6 starts in 25 min; about 1.6 h of A100 including the build, ≈ $4.
- IFS HRES scoring: CPU, network-bound.
