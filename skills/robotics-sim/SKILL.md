---
name: robotics-sim
description: Train and demo legged-robot locomotion policies on qBraid GPUs with MuJoCo Playground (MJX/Warp) + Brax PPO, benchmark them against the published Playground results, and ship a browser demo where the trained policy runs live (MuJoCo WASM + JS network) next to a three.js replay. Use for robot RL training, sim-to-sim checks, locomotion benchmarks, or "show a robot walking in the browser".
metadata:
  version: "0.3.0"
  status: "provisional"
  verified: "2026-10-01"
---

# Robotics simulation on qBraid (MuJoCo Playground)

GPU reinforcement learning for legged locomotion with the published
Playground recipe, benchmarked against the Playground technical report.

Start from the user's goal: if it is unclear whether they want the best
solution or the best quantum solution, ask (see solution-router).

## Get the scripts

The worked example (Unitree Go1 joystick) lives in `robotics/` of
https://github.com/qBraid/open-simulation-examples:

```bash
git clone --depth 1 https://github.com/qBraid/open-simulation-examples
```

All script paths below are relative to that repo. Its README has the full
reproduce block and every result.

## Environment and operational truth (verified 2026-10-01, gpu-l4, driver 595)

No qBraid environment carries this stack yet. Check `qbraid envs available`
first. Otherwise build a venv from `robotics/requirements.txt` on local disk
(`python3 -m venv env && env/bin/pip install -r requirements.txt`), and
package it with the manage-environments skill for reuse.

- **Stack:** `jax[cuda12]==0.11.2`, `playground==0.2.0`, `brax==0.14.2`, `mujoco==3.14.0`. Neither playground nor brax pins JAX, and pip takes the newest.
  - Brax 0.14.2 still calls `jax.device_put_replicated`, which was removed in JAX ≥ 0.10.
  - Flax 0.12.10 needs JAX ≥ 0.11.1, so downgrading JAX is a dead end.
  - Import the 10-line `jax_compat.py` shim before training.
- **Disk:** the venv is **6.4 GB**, 4.5 GB of it NVIDIA CUDA wheels. Put it on local `/tmp`, never on the network home of GPU instances.
- **First `registry.load()`** clones MuJoCo Menagerie (about 30 s) into site-packages.
- **Backend:** Playground 0.2.0 envs default to `impl="warp"` (MuJoCo Warp), not JAX MJX. Warp prints solver/line-search "iterations limit reached" from inside GPU kernels for every world, every step. At 8192 worlds that's **8 GB of log in 20 min** and a large slowdown. Import `warp_quiet.py`, which clears those two `opt.warn_overflow` bits at model creation (physics unchanged), and pipe stdout through `logcap.py`.
- **Throughput:** Go1JoystickFlatTerrain at 8192 envs runs at **183k PPO steps/s on one L4** with logging silenced. The report gives 417k on an A100 (Table 7, MJX). On `gpu-l4` that is about $0.07 per 100M steps at that rate, and $0.14 per 100M measured end to end including compile and evaluations.
- **Checkpoint for interruption:** instances and GPUs can disappear mid-run on cloud machines. Save `params_<step>.pkl` at every eval via `policy_params_fn`. Resume with `restore_params=` and offset the step counter (`run_resume.sh`). Normaliser stats come back with the params; the Adam state resets.

## Decision rules

- **Simulator.** MuJoCo Playground (Apache-2.0) is the default for RL-to-real recipes with published baselines. Genesis or Newton (Apache-2.0) suit very large parallel or differentiable scenes; Drake suits model-based control. Isaac Sim/Lab is free but not OSS.
- **GPU by task** (pick from `qbraid compute list`, launch via qbraid-cloud-orchestration with `--auto-stop`, terminate when done):
  - flat-terrain quadruped joystick (200M steps): `gpu-l4`, about 37 min of GPU, about $0.30 including compile and resume;
  - rough terrain or humanoids (3-4x slower per step): `gpu-a100-sxm`/`gpu-h100-sxm`, or accept multi-hour L4 runs (rough terrain plus the yaw fine-tune: about 2 h on gpu-l4, ~$1).
- **Benchmark first.** Use the same env and recipe as the report, and compare eval reward against its Figure 11 curve. Add physical metrics too: velocity-tracking RMSE and fall rate under the env's own command distribution. Reward alone hides falls, because it's clipped at 0.

## Browser demo

- **Replay:** record body `xpos`/`xquat` at the 50 Hz control rate and drive the Go1 visual meshes (geom group 2) in three.js.
- **Live mode:** use `@mujoco/mujoco@3.14.0` (official WASM) from jsDelivr.
  - Ship a **lean `.mjb`**: delete mesh geoms, meshes and textures with `MjSpec`, then re-apply the env's post-load edits (timestep, ccd_iterations, damping, PD gains). This takes 49 MB down to 47 KB.
  - **Assert** bit-identical dynamics against `env.mj_model` before shipping (`export_live.py` does it).
  - Policy in JS: normalise → MLP with swish hidden layers → `tanh` of the first `nu` outputs → `ctrl = home + 0.5·a`, with 5 substeps per action. `live_check.py` is the Python twin and must agree with Brax inference to float32 precision.
- **Agent Canvas:** the canvas is an origin-null srcdoc frame. MuJoCo WASM from jsDelivr loads there (CORS `*`); verified with a `sandbox="allow-scripts"` srcdoc wrapper in headless Chromium. Render with the agent-canvas skill.
- **Screenshots:** use Playwright's `chrome-headless-shell` with a conda-forge env holding the Chromium system libraries (atk, at-spi2, xorg libs, libxkbcommon, alsa-lib, pango) on `LD_LIBRARY_PATH`, and the flags `--use-angle=swiftshader --enable-unsafe-swiftshader --virtual-time-budget=…`. The viewer's URL hashes (`#live`, `#t=12`, `#dark`) set up states for screenshots.
- **After a time jump, snap the follow camera.** At software-rendering frame rates a lerping camera never catches up, and screenshots look wrong.

## Recipe

`train.py` (official config, seed 0) → `run_resume.sh` if interrupted → `evaluate.py` (final + early) → `export_geometry.py` → `export_live.py` → `live_check.py` (with the model solver and the default solver) → `make_bench.py` → `build_viewer.py`.

| Metric | Value |
|---|---|
| Final eval reward | **28.1**, against ≈27 in Playground Fig. 11 |
| Tracking RMSE | 0.082 m/s xy, 0.085 rad/s yaw |
| Falls | 0 of 512 twenty-second episodes |
| Sim-to-sim in CPU MuJoCo | 0.057 m/s (0.066 with MuJoCo's default solver), no fall |
| JS network vs Brax | agrees to 1e-6 |

Mid-training it lagged the reference by about 20M steps (20.3 vs ≈25 at 60M), with one seed and Adam reset at the resume. Five seeds give a proper band at 5x the cost.

## Quantum: the short answer

**No credible near-term path.** Quantum reinforcement learning and quantum
machine learning for control exist only as research at toy scale (a few
qubits, simple tasks), with no evidence of an advantage over GPU-parallel
simulation and PPO. The best answer today is the classical GPU recipe above.
Any QPU experiment for learning purposes is priced with `qbraid devices get`
and confirmed with the user first. For the full answer and the experiment
protocol, use the **quantum-readiness** skill (section "Machine learning and
control").

## Verification stamp

- 2026-10-01: jax 0.11.2 (cuda12), playground 0.2.0, brax 0.14.2, mujoco 3.14.0, flax 0.12.10.
- Instance: `gpu-l4` (NVIDIA L4 24 GB, driver 595).
- Training: 217.9M steps, 36.7 min of GPU, $0.30; all GPU use including evals and a failed start 42 min, $0.34.
- Re-verify on every JAX, Brax or Playground release: the pins above break first.
