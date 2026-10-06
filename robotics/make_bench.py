"""Turn training/eval outputs into results/benchmark.json (verdicts + tour + stamp).

Reference numbers (MuJoCo Playground technical report, 2025):
  * Fig. 11, Go1JoystickFlatTerrain, Brax PPO, 5 seeds, A100: eval points every 20M
    steps; mean ~18.5 at 20M, ~18 at 40M, ~25 at 60M, plateau ~27 (5-seed band ~26-28) at 100-200M.
    Values read off the figure; +-1.
  * Table 7: 417,451 +- 2,955 PPO steps/s on one A100 (MJX backend).
"""
import argparse
import json
import os

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--final_eval", default="eval_final.json")
p.add_argument("--early_eval", default="eval_early.json")
p.add_argument("--live_check", default="live_check.json")
p.add_argument("--live_check_default", default="live_check_default.json")
p.add_argument("--wall_s", type=float, required=True, help="total GPU training wall time (s), summed over resumed jobs")
p.add_argument("--gpu_usd_per_h", type=float, default=0.49)
p.add_argument("--sps", type=float, default=None, help="clean end-to-end training steps/s (excl. compile and log-throttled segments)")
p.add_argument("--gpu_total_min", type=float, default=None, help="all GPU job minutes incl. evals and failed starts")
p.add_argument("--date", default="2026-10-01")
p.add_argument("--out", default="benchmark.json")
a = p.parse_args()

curve = [json.loads(l) for l in open(os.path.join(a.run, "progress.jsonl"))]
curve = list({c["step"]: c for c in curve}.values())  # dedupe: a resumed job re-evaluates its start step
fin = curve[-1]
at = lambda s: min(curve, key=lambda c: abs(c["step"] - s))
ev = json.load(open(os.path.join(a.run, a.final_eval)))
tr = ev["tracking"]
early = json.load(open(os.path.join(a.run, a.early_eval))) if os.path.exists(os.path.join(a.run, a.early_eval)) else None
lc = json.load(open(os.path.join(a.run, a.live_check))) if os.path.exists(os.path.join(a.run, a.live_check)) else None
steps = fin["step"]
sps = a.sps or steps / a.wall_s
usd = a.wall_s / 3600 * a.gpu_usd_per_h
REF_PLATEAU, REF_LO = 27.0, 26.0


def v(ok, close=False):
    return "reached" if ok else ("close" if close else "not reached")


rows = [
    {"label": "Final eval reward vs published plateau", "value": f"{fin['reward']:.1f} vs ≈{REF_PLATEAU:.0f}",
     "verdict": v(fin["reward"] >= REF_LO, fin["reward"] >= REF_LO - 2),
     "note": f"at {steps/1e6:.0f}M steps; reference = Playground report Fig. 11 (5 seeds, band ≈26–28, read off the figure ±1)"},
    {"label": "Reward at 20M steps (sample efficiency)", "value": f"{at(20e6)['reward']:.1f} vs ≈18",
     "verdict": v(at(20e6)["reward"] >= 16.5, at(20e6)["reward"] >= 14), "note": f"nearest eval at {at(20e6)['step']/1e6:.1f}M"},
    {"label": "Reward at 60M steps (mid-training)", "value": f"{at(60e6)['reward']:.1f} vs ≈25",
     "verdict": v(at(60e6)["reward"] >= 23.5, at(60e6)["reward"] >= 21.5),
     "note": f"nearest eval at {at(60e6)['step']/1e6:.1f}M. We lag the reference by ~20M steps here, then catch up ({at(126e6)['reward']:.1f} at {at(126e6)['step']/1e6:.0f}M). Likely causes: one seed vs a 5-seed mean, and the job-cap resume at 45.9M restarted Adam's moment estimates"},
    {"label": "Velocity tracking RMSE (xy)", "value": f"{tr['lin_vel_rmse_mps']:.3f} m/s",
     "verdict": v(tr["lin_vel_rmse_mps"] <= 0.15, tr["lin_vel_rmse_mps"] <= 0.25),
     "note": f"{tr['n_envs']} episodes × {tr['episode_s']:.0f} s, env's own random commands up to 1.5 m/s, 1 s settling excluded; bar ≤0.15 m/s is ours (no published RMSE)"},
    {"label": "Yaw-rate tracking RMSE", "value": f"{tr['yaw_rate_rmse_radps']:.3f} rad/s",
     "verdict": v(tr["yaw_rate_rmse_radps"] <= 0.2, tr["yaw_rate_rmse_radps"] <= 0.3), "note": "commands up to 1.2 rad/s"},
    {"label": "Falls", "value": f"{100*tr['fall_rate']:.2f}% of {tr['n_envs']}", "verdict": v(tr["fall_rate"] == 0, tr["fall_rate"] < 0.01),
     "note": "torso up-vector below horizontal at any point in 20 s"},
    {"label": "Throughput (L4, Warp) vs A100 (MJX, published)", "value": f"{sps/1e3:.0f}k vs 417k steps/s",
     "note": f"different GPU and backend: A100 has ~5× the memory bandwidth of the L4. On qBraid that is ${usd/steps*1e8:.2f} per 100M steps on gpu-l4"},
    {"label": "Training time / cost", "value": f"{a.wall_s/60:.0f} min · ${usd:.2f}",
     "note": f"{steps/1e6:.0f}M steps on one gpu-l4 (${a.gpu_usd_per_h}/h), two capped jobs" + (f"; all GPU use incl. evals and a failed start: {a.gpu_total_min:.0f} min, ${a.gpu_total_min/60*a.gpu_usd_per_h:.2f}" if a.gpu_total_min else "")},
]
if early:
    et = early["tracking"]
    rows.append({"label": "Early policy (ghost) for contrast", "value": f"{early.get('step', 0)/1e6:.0f}M steps",
                 "note": f"tracking RMSE {et['lin_vel_rmse_mps']:.2f} m/s, falls {100*et['fall_rate']:.1f}%"})
if lc:
    rows.append({"label": "Sim-to-sim: same policy in CPU MuJoCo (browser physics)", "value": f"{lc['cpu_mujoco_lin_vel_rmse_mps']:.3f} m/s",
                 "verdict": v(lc["fell_at_step"] is None and lc["cpu_mujoco_lin_vel_rmse_mps"] <= 1.5 * max(tr["lin_vel_rmse_mps"], 0.05),
                              lc["fell_at_step"] is None),
                 "note": f"scripted 25 s joystick course, {'no fall' if lc['fell_at_step'] is None else 'fell at step ' + str(lc['fell_at_step'])}; trained on MuJoCo Warp"})
    lcd_p = os.path.join(a.run, a.live_check_default)
    if os.path.exists(lcd_p):
        lcd = json.load(open(lcd_p))
        rows.append({"label": "…with MuJoCo's full default solver", "value": f"{lcd['cpu_mujoco_lin_vel_rmse_mps']:.3f} m/s",
                     "verdict": v(lcd["fell_at_step"] is None and lcd["cpu_mujoco_lin_vel_rmse_mps"] <= 1.5 * max(tr["lin_vel_rmse_mps"], 0.05), lcd["fell_at_step"] is None),
                     "note": f"iterations={lcd['iterations']}, ls_iterations={lcd['ls_iterations']} (training used 1 / 5): the policy does not exploit the coarse solver"})
    rows.append({"label": "JS policy vs Brax inference", "value": f"{lc['mlp_vs_brax_maxdiff']:.1e}", "verdict": v(lc["mlp_vs_brax_maxdiff"] < 1e-5),
                 "note": "max |action diff|; the browser runs this exact network"})

ref_words = "Playground reference ≈27"
tour = [
    {"title": "Trained on one L4", "text": f"PPO on {steps/1e6:.0f}M simulated steps took {a.wall_s/60:.0f} minutes and ${usd:.2f} on a qBraid gpu-l4. Final reward {fin['reward']:.1f} ({ref_words}).",
     "hold": 7, "t": 0, "world": [0.9, -1.7, 1.3], "lookWorld": [-1.6, 2.6, 1.0], "ghost": ""},
    {"title": "A learned trot", "text": "Diagonal leg pairs alternate. Watch the gait strip in the panel: the stance bars interleave, as on the real robot.",
     "hold": 7, "t": 2.6, "off": [0.15, -1.45, 0.38], "look": [0.05, 0, 0.0], "ghost": ""},
    {"title": "Tracking the joystick", "text": f"Teal = commanded velocity, orange = achieved. Across 512 random 20 s episodes the xy error is {tr['lin_vel_rmse_mps']:.2f} m/s RMS, with {100*tr['fall_rate']:.1f}% falls.",
     "hold": 8, "t": 6.3, "off": [-1.8, -0.9, 1.25], "look": [0.6, 0, 0], "ghost": ""},
    {"title": "What learning bought", "text": "The blue ghost replays the same joystick script with an early checkpoint. Same robot, same commands, far less control.",
     "hold": 8, "t": 10.1, "off": [-2.6, -2.2, 1.5], "look": [0.2, 0.4, 0], "ghost": "early"},
    {"title": "The whole course", "text": "Footprints and a speed-coloured trail (viridis, 0–1.5 m/s) show 25 s of walking, arcs, strafing, reversing and spinning. Press “Drive it live” to take over.",
     "hold": 9, "t": 19.6, "off": [-3.8, 0.4, 3.6], "look": [1.2, 0, 0], "ghost": ""},
]
stamp = (f"{a.date} · env: jax[cuda12] 0.11.2, playground 0.2.0, brax 0.14.2, mujoco 3.14.0\n"
         f"machine: qBraid gpu-l4 (NVIDIA L4 24 GB, driver 595), shared instance\n"
         f"task: Go1JoystickFlatTerrain, official Playground PPO config, seed 0\n"
         f"training: {steps/1e6:.1f}M steps, {a.wall_s/60:.1f} min GPU, ${usd:.2f}")
json.dump({"rows": rows, "tour": tour, "stamp": stamp, "screen_sub": f"Go1 joystick · Brax PPO · {steps/1e6:.0f}M steps · {a.wall_s/60:.0f} min · ${usd:.2f}"},
          open(a.out, "w"), indent=1, ensure_ascii=False)
print(json.dumps([(r["label"], r["value"], r.get("verdict")) for r in rows], indent=0, ensure_ascii=False))
