"""GPU Monte Carlo market risk: VaR and Expected Shortfall on an NVIDIA L4.

Part A (validation): a linear 20-asset portfolio with jointly normal 10-day
P&L has closed-form VaR and ES. We estimate both by Monte Carlo at N = 1e3 .. 1e8
with 32 independent replications per N and check two things:
  * bias: the mean estimate converges to the closed form;
  * error bars: the replication standard deviation matches the asymptotic
    standard error of the quantile and ES estimators,
      SE(VaR_a) = sqrt(a(1-a)/N) / f(VaR_a)
      SE(ES_a)  = sqrt( (Var(L | L>=VaR_a) + a (ES_a - VaR_a)^2) / ((1-a) N) )

Part B (speed): full revaluation of a 500-option book (Black-Scholes) on 20
correlated underlyings over a 10-day horizon, GPU (CuPy) against CPU (NumPy,
2 threads), plus a 1,500-path daily path fan for the viewer.

Loss L is positive for losses. alpha = 0.99 for VaR (Basel 2.5), 0.975 for ES
(FRTB).
"""
import json
import math
import os
import time
from pathlib import Path

import cupy as cp
import numpy as np
from scipy.stats import norm

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(7)
N_ASSET = 20
H_DAYS = 10
A_VAR, A_ES = 0.99, 0.975


def market():
    vol = rng.uniform(0.15, 0.45, N_ASSET)                    # annual vols
    f = rng.normal(size=(N_ASSET, 3))                          # 3-factor correlation
    c = f @ f.T + np.diag(rng.uniform(0.5, 1.5, N_ASSET))
    d = np.sqrt(np.diag(c)); corr = c / np.outer(d, d)
    mu = rng.uniform(0.02, 0.10, N_ASSET)
    return vol, corr, mu


def part_a(vol, corr, mu):
    h = H_DAYS / 252
    pos = rng.uniform(-2e6, 5e6, N_ASSET)                      # $ exposures, long/short
    cov = np.outer(vol, vol) * corr * h
    m = -(pos @ (mu * h))                                      # mean loss
    s = math.sqrt(pos @ cov @ pos)
    var_cf = m + s * norm.ppf(A_VAR)
    es_cf = m + s * norm.pdf(norm.ppf(A_ES)) / (1 - A_ES)
    var975 = m + s * norm.ppf(A_ES)
    # theory for the estimator SEs (normal loss)
    f_var = norm.pdf(norm.ppf(A_VAR)) / s
    z = norm.ppf(A_ES)
    tail_var = s**2 * (1 + z * norm.pdf(z) / (1 - A_ES) - (norm.pdf(z) / (1 - A_ES)) ** 2)
    L = cp.asarray(np.linalg.cholesky(cov))
    posg = cp.asarray(pos); mug = cp.asarray(mu * h)
    rows = []
    for n in [10**k for k in range(3, 9)]:
        reps, chunk = (32 if n < 10**8 else 8), min(n, 10**7)
        v_est, e_est = [], []
        t0 = time.perf_counter()
        for r in range(reps):
            if n <= chunk:
                zz = cp.random.standard_normal((n, N_ASSET), dtype=cp.float64)
                loss = -((zz @ L.T + mug) @ posg)
                v = float(cp.quantile(loss, A_VAR)); q = float(cp.quantile(loss, A_ES))
                e = float(loss[loss >= q].mean())
            else:  # 1e8: stream in chunks, exact quantiles from a histogram refinement is overkill;
                # instead keep the top tail (losses above a conservative threshold) on device.
                thr = m + s * norm.ppf(0.96)
                tails = []
                for _ in range(n // chunk):
                    zz = cp.random.standard_normal((chunk, N_ASSET), dtype=cp.float64)
                    loss = -((zz @ L.T + mug) @ posg)
                    tails.append(loss[loss >= thr])
                tail = cp.sort(cp.concatenate(tails))[::-1]
                k_var = int(round((1 - A_VAR) * n)); k_es = int(round((1 - A_ES) * n))
                v = float(tail[k_var - 1]); e = float(tail[:k_es].mean())
            v_est.append(v); e_est.append(e)
        cp.cuda.Stream.null.synchronize()
        dt = time.perf_counter() - t0
        v_est, e_est = np.array(v_est), np.array(e_est)
        rows.append({
            "N": n, "reps": reps, "seconds_total": dt, "scen_per_s": n * reps / dt,
            "var_mean": float(v_est.mean()), "var_sd": float(v_est.std(ddof=1)),
            "var_se_theory": math.sqrt(A_VAR * (1 - A_VAR) / n) / f_var,
            "es_mean": float(e_est.mean()), "es_sd": float(e_est.std(ddof=1)),
            "es_se_theory": math.sqrt((tail_var + A_ES * (es_cf - var975) ** 2) / ((1 - A_ES) * n)),
        })
        r = rows[-1]
        print(f"N={n:>9}: VaR {r['var_mean']:,.0f} (cf {var_cf:,.0f}) sd {r['var_sd']:,.0f} vs th {r['var_se_theory']:,.0f} | "
              f"ES {r['es_mean']:,.0f} (cf {es_cf:,.0f}) sd {r['es_sd']:,.0f} vs th {r['es_se_theory']:,.0f} | {dt:.2f}s", flush=True)
    return {"var99_closed_form": var_cf, "es975_closed_form": es_cf, "loss_sd": s,
            "portfolio_gross_usd": float(np.abs(pos).sum()), "convergence": rows}


def bs_price(xp, s, k, t, r, vol, is_call):
    t = xp.maximum(t, 1e-8)
    sq = vol * xp.sqrt(t)
    d1 = (xp.log(s / k) + (r + 0.5 * vol * vol) * t) / sq
    d2 = d1 - sq
    if xp is cp:
        from cupyx.scipy.special import ndtr
    else:
        from scipy.special import ndtr
    call = s * ndtr(d1) - k * xp.exp(-r * t) * ndtr(d2)
    put = k * xp.exp(-r * t) * ndtr(-d2) - s * ndtr(-d1)
    return xp.where(is_call, call, put)


def book(vol):
    n_opt = 500
    und = rng.integers(0, N_ASSET, n_opt)
    s0 = np.full(N_ASSET, 100.0)
    k = s0[und] * rng.uniform(0.8, 1.2, n_opt)
    t = rng.choice([0.08, 0.25, 0.5, 1.0], n_opt)
    is_call = rng.random(n_opt) < 0.5
    qty = rng.integers(-200, 400, n_opt).astype(float) * 10
    return dict(und=und, s0=s0, k=k, t=t, is_call=is_call, qty=qty, vol=vol[und])


def revalue(xp, b, z, corr_l, vol, h, r=0.03):
    """Loss of the option book for scenario shocks z (n x N_ASSET)."""
    und = xp.asarray(b["und"]); k = xp.asarray(b["k"]); t = xp.asarray(b["t"])
    ic = xp.asarray(b["is_call"]); q = xp.asarray(b["qty"]); ov = xp.asarray(b["vol"])
    s0 = xp.asarray(b["s0"]); v = xp.asarray(vol)
    v0 = bs_price(xp, s0[und], k, t, r, ov, ic) @ q
    x = z @ xp.asarray(corr_l).T
    s1 = s0 * xp.exp((r - 0.5 * v * v) * h + v * math.sqrt(h) * x)
    p1 = bs_price(xp, s1[:, und], k, t - h, r, ov, ic) @ q
    return v0 - p1


def part_b(vol, corr):
    b = book(vol)
    cl = np.linalg.cholesky(corr)
    h = H_DAYS / 252
    out = {"n_options": len(b["qty"]), "n_underlyings": N_ASSET}
    # GPU timing
    n_gpu = 2_000_000
    revalue(cp, b, cp.random.standard_normal((1000, N_ASSET)), cl, vol, h)  # warm-up / JIT
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter(); losses = []
    for _ in range(5):
        losses.append(revalue(cp, b, cp.random.standard_normal((n_gpu // 5, N_ASSET)), cl, vol, h))
    loss = cp.concatenate(losses); cp.cuda.Stream.null.synchronize()
    tg = time.perf_counter() - t0
    var = float(cp.quantile(loss, A_VAR)); q = float(cp.quantile(loss, A_ES)); es = float(loss[loss >= q].mean())
    # CPU timing on a smaller N (2 threads, the per-worker allowance on the pool box)
    n_cpu = 100_000
    t0 = time.perf_counter()
    revalue(np, b, rng.standard_normal((n_cpu, N_ASSET)), cl, vol, h)
    tc = time.perf_counter() - t0
    out.update({"gpu_scenarios": n_gpu, "gpu_seconds": tg, "gpu_scen_per_s": n_gpu / tg,
                "cpu_scenarios": n_cpu, "cpu_seconds": tc, "cpu_scen_per_s": n_cpu / tc,
                "cpu_threads": int(os.environ.get("OMP_NUM_THREADS", "0") or 0),
                "speedup": (n_gpu / tg) / (n_cpu / tc), "var99": var, "es975": es})
    hist, edges = np.histogram(cp.asnumpy(loss), bins=160)
    out["loss_hist"] = {"counts": hist.tolist(), "edges": edges.tolist()}
    # path fan: 1,500 paths revalued daily over the horizon (shared shocks per path)
    n_paths, days = 1500, H_DAYS
    zs = rng.standard_normal((days, n_paths, N_ASSET))
    cum = np.cumsum(zs, axis=0)
    fan = []
    for d in range(1, days + 1):
        x = cum[d - 1] / math.sqrt(d)                           # N(0,1) shocks at horizon d
        fan.append((-revalue(np, b, x, cl, vol, d / 252)).round(0).tolist())  # P&L (gain positive)
    pnl_T = np.array(fan[-1])
    out["path_fan"] = {"days": days, "pnl": fan, "var99_paths": float(-np.quantile(pnl_T, 1 - A_VAR))}
    print(f"GPU {n_gpu/tg:,.0f} scen/s vs CPU {n_cpu/tc:,.0f} scen/s -> {out['speedup']:.0f}x; VaR99 {var:,.0f} ES97.5 {es:,.0f}")
    return out


def main():
    vol, corr, mu = market()
    dev = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    t0 = time.time()
    a = part_a(vol, corr, mu)
    b = part_b(vol, corr)
    res = {"device": dev, "cupy": cp.__version__, "horizon_days": H_DAYS, "alpha_var": A_VAR, "alpha_es": A_ES,
           "linear_validation": a, "option_book": b, "wall_s": time.time() - t0}
    (OUT / "mc_risk.json").write_text(json.dumps(res))
    print(f"wall {res['wall_s']:.0f}s on {dev}")


if __name__ == "__main__":
    main()
