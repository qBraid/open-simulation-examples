"""Export everything the in-browser live mode needs (MuJoCo WASM + JS policy).

Writes live.json with: the compiled Go1 model (.mjb, base64), the observation
recipe (sensor addresses, IMU site, home pose), control constants, and the
trained policy: observation normaliser (mean, std) and MLP weights. The browser
runs MuJoCo 3.14 (@mujoco/mujoco), rebuilds the 48-d 'state' observation exactly
as mujoco_playground's Go1 joystick env does (without sensor noise), evaluates
the MLP (swish hidden layers, tanh on the mean) and applies position targets.

    python export_live.py --run runs/go1_flat --params params_final.pkl --out live.json
"""
import argparse
import base64
import json
import os
import pickle
import tempfile

import mujoco
import numpy as np
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params

p = argparse.ArgumentParser()
p.add_argument("--env", default="Go1JoystickFlatTerrain")
p.add_argument("--run", required=True)
p.add_argument("--params", default="params_final.pkl")
p.add_argument("--out", default="live.json")
a = p.parse_args()

env = registry.load(a.env)
m = env.mj_model
cfg = registry.get_default_config(a.env)
net = dict(locomotion_params.brax_ppo_config(a.env).network_factory)

# Lean physics copy: drop visual meshes and textures (the viewer draws its own
# meshes), then re-apply the env's post-load modifications and prove the
# dynamics are identical to env.mj_model by rolling both out under the same
# control sequence.
from etils import epath
from mujoco_playground._src.locomotion.go1 import base as go1_base

spec = mujoco.MjSpec.from_file(str(env._xml_path), assets=go1_base.get_assets())
for g in list(spec.geoms):
    if g.type == mujoco.mjtGeom.mjGEOM_MESH:
        spec.delete(g)
for x in list(spec.meshes) + list(spec.textures):
    spec.delete(x)
for mat in spec.materials:
    mat.textures = [""] * len(mat.textures)
lean = spec.compile()
lean.opt.timestep = cfg.sim_dt
lean.opt.ccd_iterations = 20
lean.dof_damping[6:] = cfg.Kd
lean.actuator_gainprm[:, 0] = cfg.Kp
lean.actuator_biasprm[:, 1] = -cfg.Kp
assert lean.nbody == m.nbody and lean.nu == m.nu and lean.nsensor == m.nsensor
for f in ("body_mass", "body_inertia", "body_ipos", "dof_damping", "dof_armature", "actuator_gainprm", "actuator_biasprm", "jnt_range"):
    assert np.allclose(getattr(lean, f), getattr(m, f)), f
d0, d1 = mujoco.MjData(m), mujoco.MjData(lean)
mujoco.mj_resetDataKeyframe(m, d0, m.keyframe("home").id)
mujoco.mj_resetDataKeyframe(lean, d1, lean.keyframe("home").id)
rng = np.random.default_rng(0)
maxdiff = 0.0
for t in range(500):
    u = home_ctrl = m.keyframe("home").qpos[7:] + 0.3 * np.sin(0.05 * t + np.arange(m.nu))
    d0.ctrl[:] = u; d1.ctrl[:] = u
    mujoco.mj_step(m, d0); mujoco.mj_step(lean, d1)
    maxdiff = max(maxdiff, float(np.abs(d0.qpos - d1.qpos).max()), float(np.abs(d0.sensordata - d1.sensordata).max()))
print("lean-vs-env max |qpos/sensor diff| over 500 steps:", maxdiff)
assert maxdiff < 1e-9
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "go1.mjb")
    mujoco.mj_saveModel(lean, path, None)
    mjb = open(path, "rb").read()
m = lean


def sensor_adr(name):
    s = m.sensor(name)
    return int(m.sensor_adr[s.id]), int(m.sensor_dim[s.id])


home = m.keyframe("home")
with open(os.path.join(a.run, a.params), "rb") as f:
    params = pickle.load(f)
norm, pol = params[0], params[1]
key = net.get("policy_obs_key", "state")
mean = np.asarray(norm.mean[key] if isinstance(norm.mean, dict) else norm.mean, np.float32)
std = np.asarray(norm.std[key] if isinstance(norm.std, dict) else norm.std, np.float32)
layers = []
pp = pol["params"]
for name in sorted(pp, key=lambda s: int(s.split("_")[-1])):
    W = np.asarray(pp[name]["kernel"], np.float32)
    b = np.asarray(pp[name]["bias"], np.float32)
    layers.append({"in": W.shape[0], "out": W.shape[1],
                   "W": base64.b64encode(W.tobytes()).decode(), "b": base64.b64encode(b.tobytes()).decode()})
out = {
    "mjb": base64.b64encode(mjb).decode(), "mjb_bytes": len(mjb),
    "nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu), "nbody": int(m.nbody),
    "timestep": float(m.opt.timestep), "n_substeps": int(round(cfg.ctrl_dt / cfg.sim_dt)), "ctrl_dt": float(cfg.ctrl_dt),
    "action_scale": float(cfg.action_scale), "home_qpos": home.qpos.tolist(), "default_pose": home.qpos[7:].tolist(),
    "home_key": int(home.id),
    "obs": {"local_linvel": sensor_adr("local_linvel"), "gyro": sensor_adr("gyro"),
            "imu_site": int(m.site("imu").id), "upvector": sensor_adr("upvector")},
    "cmd_range": list(cfg.command_config.a),
    "lean_equivalence_maxdiff": maxdiff,
    "policy": {"obs_key": key, "obs_dim": int(mean.shape[0]), "mean": mean.tolist(), "std": std.tolist(),
               "activation": "swish", "layers": layers, "act_dim": int(m.nu)},
}
json.dump(out, open(a.out, "w"))
print(a.out, f"{os.path.getsize(a.out)/1e6:.2f} MB", "mjb", len(mjb), "layers", [(l['in'], l['out']) for l in layers], "obs", mean.shape)
