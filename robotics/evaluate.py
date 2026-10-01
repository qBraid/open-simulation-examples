"""Evaluate a trained Go1 joystick policy and record rollouts for the viewer.

1. Tracking benchmark: N parallel episodes (20 s each) with the environment's own
   random joystick commands. Reports RMS error of base xy velocity vs command and
   of yaw rate vs command (1 s settling window after each command change is
   excluded), plus the fall rate (torso up-vector pointing down).
2. Scripted demo rollout: a fixed joystick script (stand, walk, arc, strafe, back
   up, spin, sprint, stop) recording every body's world pose at the 50 Hz
   control rate, for the three.js replay.

    python evaluate.py --run runs/go1_flat --params params_final.pkl --tag final
"""
import argparse
import functools
import json
import os
import pickle
import time

import jax
import jax_compat  # noqa: F401  (Brax 0.14 on JAX >= 0.11)
import warp_quiet  # noqa: F401  (no per-world line-search printf)
import jax.numpy as jp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params

p = argparse.ArgumentParser()
p.add_argument("--env", default="Go1JoystickFlatTerrain")
p.add_argument("--run", required=True)
p.add_argument("--params", default="params_final.pkl")
p.add_argument("--tag", default="final")
p.add_argument("--n_envs", type=int, default=512)
p.add_argument("--steps", type=int, default=1000)
p.add_argument("--settle", type=int, default=50, help="control steps ignored after a command change")
p.add_argument("--no_track", action="store_true")
a = p.parse_args()

env = registry.load(a.env)
ppo_params = locomotion_params.brax_ppo_config(a.env)
net_kw = dict(ppo_params.network_factory)
nets = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size,
    preprocess_observations_fn=running_statistics.normalize, **net_kw)
make_inference = ppo_networks.make_inference_fn(nets)
with open(os.path.join(a.run, a.params), "rb") as f:
    params = pickle.load(f)
policy = jax.jit(make_inference(params[:2], deterministic=True))
out = {"env": a.env, "params": a.params, "tag": a.tag}

# ---------------------------------------------------------------- tracking benchmark
if not a.no_track:
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    linvel = jax.jit(jax.vmap(env.get_local_linvel))
    gyro = jax.jit(jax.vmap(env.get_gyro))
    upz = jax.jit(jax.vmap(lambda d: env.get_upvector(d)[-1]))
    key = jax.random.PRNGKey(123)
    state = reset(jax.random.split(key, a.n_envs))
    alive = np.ones(a.n_envs, bool)
    since_change = np.zeros(a.n_envs, int)
    prev_cmd = np.asarray(state.info["command"])
    se_v, se_w, n_v = 0.0, 0.0, 0
    speeds_cmd, speeds_act = [], []
    t0 = time.time()
    for t in range(a.steps):
        key, k = jax.random.split(key)
        act, _ = policy(state.obs, jax.random.split(k, a.n_envs))
        state = step(state, act)
        cmd_used = prev_cmd  # obs at this step carried prev command
        v = np.asarray(linvel(state.data))[:, :2]
        w = np.asarray(gyro(state.data))[:, 2]
        fell = np.asarray(upz(state.data)) < 0.0
        alive &= ~fell
        m = alive & (since_change >= a.settle)
        if m.any():
            se_v += float(np.sum(np.sum((v[m] - cmd_used[m, :2]) ** 2, axis=1)))
            se_w += float(np.sum((w[m] - cmd_used[m, 2]) ** 2))
            n_v += int(m.sum())
            if t % 10 == 0:
                speeds_cmd.extend(np.linalg.norm(cmd_used[m, :2], axis=1)[:64].tolist())
                speeds_act.extend(np.linalg.norm(v[m], axis=1)[:64].tolist())
        cmd = np.asarray(state.info["command"])
        changed = np.any(np.abs(cmd - prev_cmd) > 1e-6, axis=1)
        since_change = np.where(changed, 0, since_change + 1)
        prev_cmd = cmd
    out["tracking"] = {
        "n_envs": a.n_envs, "episode_s": a.steps * env.dt, "settle_s": a.settle * env.dt,
        "lin_vel_rmse_mps": (se_v / max(n_v, 1)) ** 0.5,
        "yaw_rate_rmse_radps": (se_w / max(n_v, 1)) ** 0.5,
        "fall_rate": float(1 - alive.mean()),
        "samples": n_v, "wall_s": round(time.time() - t0, 1),
        "cmd_vs_actual_speed": [speeds_cmd[:600], speeds_act[:600]],
    }
    print(json.dumps({k: v for k, v in out["tracking"].items() if k != "cmd_vs_actual_speed"}), flush=True)

# ---------------------------------------------------------------- scripted demo rollout
SCRIPT = [  # (seconds, [vx, vy, yaw_rate], label)
    (2.0, [0.0, 0.0, 0.0], "stand"),
    (4.0, [1.0, 0.0, 0.0], "walk forward 1.0 m/s"),
    (4.0, [0.8, 0.0, 0.7], "arc left"),
    (3.0, [0.0, 0.6, 0.0], "strafe left 0.6 m/s"),
    (3.0, [-0.7, 0.0, 0.0], "back up"),
    (3.0, [0.0, 0.0, -1.1], "spin right"),
    (4.0, [1.4, 0.0, -0.4], "fast arc right 1.4 m/s"),
    (2.0, [0.0, 0.0, 0.0], "stop"),
]
mj = env.mj_model
reset1 = jax.jit(env.reset)
step1 = jax.jit(env.step)
state = reset1(jax.random.PRNGKey(7))
frames, cmds, labels, linv = [], [], [], []
key = jax.random.PRNGKey(0)
fell_at = None
for seg_i, (dur, c, lab) in enumerate(SCRIPT):
    for _ in range(int(round(dur / env.dt))):
        info = dict(state.info)
        info["command"] = jp.asarray(c, dtype=jp.float32)
        info["steps_until_next_cmd"] = jp.asarray(10 ** 6, dtype=jp.int32)
        state = state.replace(info=info)
        key, k = jax.random.split(key)
        act, _ = policy(state.obs, k)
        state = step1(state, act)
        d = state.data
        frames.append(np.concatenate([np.asarray(d.xpos).ravel(), np.asarray(d.xquat).ravel()]).astype(np.float32))
        cmds.append(c)
        labels.append(seg_i)
        linv.append(np.asarray(env.get_local_linvel(d))[:3].tolist())
        if fell_at is None and float(env.get_upvector(d)[-1]) < 0:
            fell_at = len(frames)
out["rollout"] = {
    "dt": float(env.dt), "nbody": int(mj.nbody), "n_frames": len(frames),
    "script": [{"t": float(sum(s[0] for s in SCRIPT[:i])), "dur": s[0], "cmd": s[1], "label": s[2]} for i, s in enumerate(SCRIPT)],
    "fell_at_frame": fell_at,
}
np.savez_compressed(os.path.join(a.run, f"rollout_{a.tag}.npz"),
                    frames=np.stack(frames), cmds=np.asarray(cmds, np.float32),
                    seg=np.asarray(labels, np.int16), linvel=np.asarray(linv, np.float32))
with open(os.path.join(a.run, f"eval_{a.tag}.json"), "w") as f:
    json.dump(out, f)
print("rollout frames", len(frames), "fell_at", fell_at, flush=True)
