"""Reference twin of the browser live mode, in plain MuJoCo + numpy.

Runs exactly the computation viewer.html performs in the browser (lean .mjb
model, hand-built 48-d observation without noise, numpy MLP with swish hidden
layers and tanh(mean)), then:
  1. checks the numpy MLP against Brax's own deterministic inference on the
     same observations (should agree to float32 precision), and
  2. measures sim-to-sim transfer: the policy was trained on the MuJoCo Warp
     backend; here it drives CPU MuJoCo through a joystick script.

    python live_check.py --live live.json --run runs/go1_flat --params params_final.pkl
"""
import argparse
import base64
import json
import os
import pickle
import tempfile

import mujoco
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--live", required=True)
p.add_argument("--run", required=True)
p.add_argument("--params", default="params_final.pkl")
p.add_argument("--out", default=None)
a = p.parse_args()
L = json.load(open(a.live))
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "go1.mjb")
    open(path, "wb").write(base64.b64decode(L["mjb"]))
    m = mujoco.MjModel.from_binary_path(path)
dta = mujoco.MjData(m)
P = L["policy"]
f32 = lambda s: np.frombuffer(base64.b64decode(s), np.float32)
Ws = [f32(l["W"]).reshape(l["in"], l["out"]) for l in P["layers"]]
bs = [f32(l["b"]) for l in P["layers"]]
mean, std = np.asarray(P["mean"], np.float32), np.asarray(P["std"], np.float32)
default = np.asarray(L["default_pose"])


def mlp(obs):
    x = (obs.astype(np.float32) - mean) / std
    for i, (W, b) in enumerate(zip(Ws, bs)):
        x = x @ W + b
        if i < len(Ws) - 1:
            x = x * (1.0 / (1.0 + np.exp(-x)))  # swish
    return np.tanh(x[: L["nu"]])


def obs_of(last_act, cmd):
    o = L["obs"]
    lv = dta.sensordata[o["local_linvel"][0]:o["local_linvel"][0] + 3]
    gy = dta.sensordata[o["gyro"][0]:o["gyro"][0] + 3]
    R = dta.site_xmat[o["imu_site"]].reshape(3, 3)
    grav = R.T @ np.array([0, 0, -1.0])
    return np.concatenate([lv, gy, grav, dta.qpos[7:] - default, dta.qvel[6:], last_act, cmd])


# 1. MLP vs Brax inference on random-ish observations
import jax  # noqa: E402
import jax_compat  # noqa: E402,F401
from brax.training.acme import running_statistics  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from mujoco_playground import registry  # noqa: E402
from mujoco_playground.config import locomotion_params  # noqa: E402

env = registry.load("Go1JoystickFlatTerrain")
net_kw = dict(locomotion_params.brax_ppo_config("Go1JoystickFlatTerrain").network_factory)
nets = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                      preprocess_observations_fn=running_statistics.normalize, **net_kw)
params = pickle.load(open(os.path.join(a.run, a.params), "rb"))
brax_policy = jax.jit(ppo_networks.make_inference_fn(nets)(params[:2], deterministic=True))
rng = np.random.default_rng(0)
priv_dim = env.observation_size["privileged_state"][0] if isinstance(env.observation_size["privileged_state"], tuple) else env.observation_size["privileged_state"]
errs = []
for _ in range(32):
    s = rng.normal(size=P["obs_dim"]).astype(np.float32) * std + mean
    obs = {"state": jax.numpy.asarray(s), "privileged_state": jax.numpy.zeros(priv_dim)}
    act_b, _ = brax_policy(obs, jax.random.PRNGKey(0))
    errs.append(float(np.abs(np.asarray(act_b) - mlp(s)).max()))
mlp_err = max(errs)
print("numpy MLP vs brax inference, max |diff|:", mlp_err)

# 2. sim-to-sim: same joystick script as evaluate.py, plain MuJoCo
SCRIPT = [(2.0, [0, 0, 0]), (4.0, [1.0, 0, 0]), (4.0, [0.8, 0, 0.7]), (3.0, [0, 0.6, 0]),
          (3.0, [-0.7, 0, 0]), (3.0, [0, 0, -1.1]), (4.0, [1.4, 0, -0.4]), (2.0, [0, 0, 0])]
mujoco.mj_resetDataKeyframe(m, dta, L["home_key"])
mujoco.mj_forward(m, dta)
last = np.zeros(L["nu"])
se_v = se_w = 0.0; n = 0; fell = None; k = 0
for dur, cmd in SCRIPT:
    cmd = np.asarray(cmd, float)
    for i in range(int(round(dur / L["ctrl_dt"]))):
        act = mlp(obs_of(last, cmd))
        dta.ctrl[:] = default + act * L["action_scale"]
        for _ in range(L["n_substeps"]):
            mujoco.mj_step(m, dta)
        last = act; k += 1
        up = dta.sensordata[L["obs"]["upvector"][0] + 2]
        if fell is None and up < 0:
            fell = k
        if i >= 50:  # same 1 s settling window as evaluate.py
            lv = dta.sensordata[L["obs"]["local_linvel"][0]:L["obs"]["local_linvel"][0] + 2]
            w = dta.sensordata[L["obs"]["gyro"][0] + 2]
            se_v += float(np.sum((lv - cmd[:2]) ** 2)); se_w += float((w - cmd[2]) ** 2); n += 1
res = {"mlp_vs_brax_maxdiff": mlp_err, "cpu_mujoco_lin_vel_rmse_mps": (se_v / n) ** 0.5,
       "cpu_mujoco_yaw_rate_rmse_radps": (se_w / n) ** 0.5, "fell_at_step": fell, "steps": k}
print(json.dumps(res))
if a.out:
    json.dump(res, open(a.out, "w"))
