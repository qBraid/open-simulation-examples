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

__RESULTS__

## Reproduce
```bash
# environment (see skills/earth-sim for the pins and why they are needed)
python -m venv env && env/bin/pip install uv
env/bin/uv pip install --python env/bin/python "earth2studio[sfno]==0.19.0" \
  "makani @ git+https://github.com/NVIDIA/makani.git@b38fcb2799d7dbc146fa60459f3f9823394a8bf1" \
  "torch==2.11.*" "torch-harmonics==0.9.2" "torchvision==0.26.*" scipy
export EARTH2STUDIO_CACHE=/big/disk/e2s-cache
python forecast_score.py sfno results/          # 12 starts in 2020, GPU, about __GPU_MIN__ min on an L4
python hres_score.py results/                   # IFS HRES on the same starts (CPU, network-bound)
python make_verdict.py results/ results/sfno_viz_2020082400.npz
python build_viewer.py results/ results/sfno_viz_2020082400.npz
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
