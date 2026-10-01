"""Train a Go1 joystick locomotion policy with MuJoCo Playground (MJX) + Brax PPO.

Uses the official Playground recipe for the chosen environment (env config,
PPO hyper-parameters and domain randomisation) so results are comparable with
the MuJoCo Playground report. Logs eval reward vs steps and wall time, saves the
policy (normaliser + network params) at every evaluation, and writes Orbax
checkpoints so a run can resume across capped GPU jobs.

    python train.py --env Go1JoystickFlatTerrain --run runs/go1_flat [--steps 200_000_000]
"""
import argparse
import functools
import json
import os
import pickle
import time

import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params

p = argparse.ArgumentParser()
p.add_argument("--env", default="Go1JoystickFlatTerrain")
p.add_argument("--run", required=True)
p.add_argument("--steps", type=int, default=None, help="override num_timesteps")
p.add_argument("--num_evals", type=int, default=21)
p.add_argument("--restore", default=None, help="orbax checkpoint dir to resume from")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

os.makedirs(a.run, exist_ok=True)
env_cfg = registry.get_default_config(a.env)
env = registry.load(a.env, config=env_cfg)
ppo_params = locomotion_params.brax_ppo_config(a.env)
if a.steps:
    ppo_params.num_timesteps = a.steps
ppo_params.num_evals = a.num_evals

net_kw = dict(ppo_params.get("network_factory", {}))
ppo_kw = dict(ppo_params)
del ppo_kw["network_factory"]
network_factory = functools.partial(ppo_networks.make_ppo_networks, **net_kw)
randomizer = registry.get_domain_randomizer(a.env)

log_path = os.path.join(a.run, "progress.jsonl")
t_start = time.time()
times = [t_start]
print(f"devices={jax.devices()} steps={ppo_params.num_timesteps} envs={ppo_params.num_envs}", flush=True)


def progress(step, metrics):
    times.append(time.time())
    rec = {
        "step": int(step),
        "wall_s": round(times[-1] - t_start, 1),
        "reward": float(metrics.get("eval/episode_reward", float("nan"))),
        "reward_std": float(metrics.get("eval/episode_reward_std", float("nan"))),
        "sps": float(metrics.get("training/sps", float("nan"))) if "training/sps" in metrics else None,
    }
    for k, v in metrics.items():
        if k.startswith("eval/episode_reward/") or k in ("eval/avg_episode_length",):
            rec[k.split("/")[-1]] = float(v)
    with open(log_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec[k] for k in ("step", "wall_s", "reward", "reward_std", "sps")}), flush=True)


def save_params(step, make_policy, params):
    with open(os.path.join(a.run, f"params_{int(step):012d}.pkl"), "wb") as f:
        pickle.dump(jax.device_get(params), f)


make_inference_fn, params, metrics = ppo.train(
    environment=env,
    eval_env=registry.load(a.env, config=env_cfg),
    wrap_env_fn=wrapper.wrap_for_brax_training,
    network_factory=network_factory,
    randomization_fn=randomizer,
    progress_fn=progress,
    policy_params_fn=save_params,
    save_checkpoint_path=os.path.join(os.path.abspath(a.run), "ckpt"),
    restore_checkpoint_path=a.restore,
    seed=a.seed,
    **ppo_kw,
)
with open(os.path.join(a.run, "params_final.pkl"), "wb") as f:
    pickle.dump(jax.device_get(params), f)
with open(os.path.join(a.run, "summary.json"), "w") as f:
    json.dump({"env": a.env, "num_timesteps": int(ppo_params.num_timesteps), "num_envs": int(ppo_params.num_envs),
               "wall_s": round(time.time() - t_start, 1), "device": str(jax.devices()[0]),
               "ppo": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in ppo_kw.items()},
               "network": {k: list(v) if isinstance(v, tuple) else v for k, v in net_kw.items()}}, f, indent=1)
print("done", round(time.time() - t_start, 1), "s", flush=True)
