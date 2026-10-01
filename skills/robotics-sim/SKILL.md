---
name: robotics-sim
description: Train and demo legged-robot locomotion policies on qBraid GPUs with MuJoCo Playground (MJX/Warp) + Brax PPO, benchmark them against the published Playground results, and ship a browser demo where the trained policy runs live (MuJoCo WASM + JS network) next to a three.js replay. Use for robot RL training, sim-to-sim checks, locomotion benchmarks, or "show a robot walking in the browser".
---

# Robotics simulation on qBraid (MuJoCo Playground)

Reference implementation: `robotics/` in qBraid/open-simulation-examples (Go1 joystick).

## Operational truth (verified 2026-10-01, qBraid gpu-l4, driver 595)
- **Stack:** `jax[cuda12]==0.11.2`, `playground==0.2.0`, `brax==0.14.2`, `mujoco==3.14.0`. Neither playground nor brax pins JAX, and pip takes the newest.
  - Brax 0.14.2 still calls `jax.device_put_replicated`, which was removed in JAX ≥ 0.10.
  - Flax 0.12.10 needs JAX ≥ 0.11.1, so downgrading JAX is a dead end.
  - Import the 10-line `jax_compat.py` shim before training.
- **Disk:** the venv is **6.4 GB**, 4.5 GB of it NVIDIA CUDA wheels. Put it on local `/tmp`, never on the network home of GPU instances.
- **First `registry.load()`** clones MuJoCo Menagerie (about 30 s) into site-packages.
- **Backend:** Playground 0.2.0 envs default to `impl="warp"` (MuJoCo Warp), not JAX MJX. Linesearch-iteration warnings are harmless spam.
- **Throughput:** Go1JoystickFlatTerrain at 8192 envs runs at about **40k PPO steps/s on one L4** at 100% GPU. The report gives 417k on an A100 (Table 7, MJX era). Budget accordingly.
- **Capped GPU jobs:** save `params_<step>.pkl` at every eval via `policy_params_fn`. Resume with `restore_params=` and offset the step counter. Normaliser stats come back with the params; the Adam state resets.

## Decision rules
- **Simulator.** MuJoCo Playground (Apache-2.0) is the default for RL-to-real recipes with published baselines. Genesis or Newton (Apache-2.0) suit very large parallel or differentiable scenes; Drake suits model-based control. Isaac Sim/Lab is free but not OSS.
- **GPU by task:**
  - flat-terrain quadruped joystick (200M steps): L4, about 85 min, under $1;
  - rough terrain or humanoids (3–4x slower per step): A100/H100, or accept multi-hour L4 runs.
- **Benchmark first.** Use the same env and recipe as the report, and compare eval reward against its Figure 11 curve. Add physical metrics too: velocity-tracking RMSE and fall rate under the env's own command distribution. Reward alone hides falls, because it's clipped at 0.

## Browser demo
- **Replay:** record body `xpos`/`xquat` at the 50 Hz control rate and drive the Go1 visual meshes (geom group 2) in three.js.
- **Live mode:** use `@mujoco/mujoco@3.14.0` (official WASM) from jsDelivr.
  - Ship a **lean `.mjb`**: delete mesh geoms, meshes and textures with `MjSpec`, then re-apply the env's post-load edits (timestep, ccd_iterations, damping, PD gains). This takes 49 MB down to 47 KB.
  - **Assert** bit-identical dynamics against `env.mj_model` before shipping (`export_live.py` does it).
  - Policy in JS: normalise → MLP with swish hidden layers → `tanh` of the first `nu` outputs → `ctrl = home + 0.5·a`, with 5 substeps per action. `live_check.py` is the Python twin and must agree with Brax inference to float32 precision.
- **Screenshots:** use Playwright's `chrome-headless-shell` with the conda-forge libs (`LD_LIBRARY_PATH=/tmp/ose-envs/chromelibs/lib`, flags `--use-angle=swiftshader --enable-unsafe-swiftshader --virtual-time-budget=…`). The viewer's URL hashes (`#live`, `#t=12`, `#dark`) set up states for screenshots.

## Verified recipe
See `robotics/README.md` for the commands, numbers and stamp.
