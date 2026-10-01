"""Walk-forward out-of-sample portfolio backtest on Ken French 30 industry portfolios.

Protocol follows DeMiguel, Garlappi & Uppal (2009, RFS 22(5)), "Optimal Versus
Naive Diversification": rolling estimation window of M=120 months, monthly
rebalancing, weights at month t use only returns up to t-1 (no look-ahead),
excess returns over the 1-month T-bill, proportional transaction costs charged
on actual turnover (including drift of weights between rebalances).

Strategies (all long-only, fully invested):
  1/N            equal weight, the benchmark DGU found hard to beat
  minvar         minimum variance, sample covariance
  minvar_lw      minimum variance, Ledoit-Wolf (2004) shrinkage to scaled identity
  meanvar        maximum Sharpe ratio with sample moments (the classic "estimation error" loser)
  maxsharpe_lw   maximum Sharpe ratio, LW covariance + grand-mean shrinkage of expected returns
  erc            equal risk contribution (risk parity), LW covariance

Statistics: annualised excess return, volatility, Sharpe ratio, certainty
equivalent (gamma=1, as DGU), mean monthly turnover, max drawdown, and the
Jobson-Korkie test with Memmel (2003) correction for Sharpe(strategy) vs Sharpe(1/N).
"""
import json
import math
import sys
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "results"
M = 120
START = "1963-07"
COSTS_BP = (10, 50)


def read_french(path, marker):
    lines = Path(path).read_text().splitlines()
    i = next(j for j, l in enumerate(lines) if l.strip().startswith(marker))
    header = lines[i + 1].split(",")
    rows = []
    for l in lines[i + 2:]:
        parts = l.split(",")
        if len(parts) < 2 or not parts[0].strip().isdigit() or len(parts[0].strip()) != 6:
            break
        rows.append(parts)
    df = pd.DataFrame(rows, columns=["date"] + [h.strip() for h in header[1:]])
    df["date"] = pd.PeriodIndex(df["date"].str.strip(), freq="M")
    df = df.set_index("date").astype(float) / 100.0
    return df


def load():
    ind = read_french(DATA / "30_Industry_Portfolios.csv", "Average Value Weighted Returns -- Monthly")
    lines = (DATA / "F-F_Research_Data_Factors.csv").read_text().splitlines()
    i = next(j for j, l in enumerate(lines) if l.startswith(",Mkt-RF"))
    rows = []
    for l in lines[i + 1:]:
        p = l.split(",")
        if len(p) < 5 or len(p[0].strip()) != 6:
            break
        rows.append(p)
    ff = pd.DataFrame(rows, columns=["date", "MktRF", "SMB", "HML", "RF"])
    ff["date"] = pd.PeriodIndex(ff["date"].str.strip(), freq="M")
    ff = ff.set_index("date").astype(float) / 100.0
    ind = ind[ind.index >= pd.Period(START, "M")]
    ind = ind.replace(-0.9999, np.nan).dropna()
    rf = ff["RF"].reindex(ind.index)
    return ind, rf


def ledoit_wolf(x):
    """Ledoit & Wolf (2004) shrinkage of the sample covariance toward mu*I."""
    t, n = x.shape
    xc = x - x.mean(0)
    s = xc.T @ xc / t
    mu = np.trace(s) / n
    f = mu * np.eye(n)
    d2 = np.sum((s - f) ** 2)
    b2 = sum(np.sum((np.outer(r, r) - s) ** 2) for r in xc) / t ** 2
    b2 = min(b2, d2)
    delta = b2 / d2 if d2 > 0 else 0.0
    return delta * f + (1 - delta) * s, delta


def w_minvar(cov):
    n = cov.shape[0]
    w = cp.Variable(n)
    cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))), [cp.sum(w) == 1, w >= 0]).solve(solver=cp.CLARABEL)
    return np.clip(w.value, 0, None) / np.clip(w.value, 0, None).sum()


def w_maxsharpe(mu, cov):
    """Long-only tangency via the homogenised QP: min y'Σy s.t. μ'y = 1, y >= 0."""
    n = cov.shape[0]
    if mu.max() <= 0:
        return w_minvar(cov)
    y = cp.Variable(n)
    cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(cov))), [mu @ y == 1, y >= 0]).solve(solver=cp.CLARABEL)
    v = np.clip(y.value, 0, None)
    return v / v.sum()


def w_erc(cov, iters=500):
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(iters):  # simple fixed-point (Roncalli) iteration
        rc = w * (cov @ w)
        w_new = w * (rc.mean() / rc) ** 0.5
        w_new /= w_new.sum()
        if np.max(np.abs(w_new - w)) < 1e-10:
            break
        w = w_new
    return w


def frontier(mu, cov, npts=36):
    """Long-only efficient frontier (annualised vol, ret) from monthly moments."""
    n = len(mu)
    w = cp.Variable(n)
    target = cp.Parameter()
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))),
                      [cp.sum(w) == 1, w >= 0, mu @ w >= target])
    wmv = w_minvar(cov)
    lo, hi = float(mu @ wmv), float(mu.max())
    pts = []
    for tgt in np.linspace(lo, hi, npts):
        target.value = tgt
        prob.solve(solver=cp.CLARABEL)
        if w.value is None:
            continue
        v = np.clip(w.value, 0, None); v /= v.sum()
        pts.append([float(np.sqrt(v @ cov @ v * 12)), float(mu @ v * 12)])
    return pts


def jk_memmel(r1, r2):
    """Jobson-Korkie test (Memmel 2003 correction) of H0: SR1 == SR2. Returns z, p (two-sided)."""
    t = len(r1)
    m1, m2 = r1.mean(), r2.mean()
    s1, s2 = r1.std(ddof=1), r2.std(ddof=1)
    s12 = np.cov(r1, r2)[0, 1]
    theta = (2 * s1**2 * s2**2 - 2 * s1 * s2 * s12 + 0.5 * m1**2 * s2**2 + 0.5 * m2**2 * s1**2
             - (m1 * m2 / (s1 * s2)) * s12**2) / t
    z = (s2 * m1 - s1 * m2) / math.sqrt(theta)
    return float(z), float(2 * (1 - norm.cdf(abs(z))))


def max_drawdown(r):
    wealth = np.cumprod(1 + r)
    peak = np.maximum.accumulate(wealth)
    dd = wealth / peak - 1
    return float(dd.min()), dd


def run():
    t0 = time.time()
    ind, rf = load()
    R = ind.values
    ex = (ind.sub(rf, axis=0)).values
    dates = ind.index
    T, n = R.shape
    names = ["1/N", "minvar", "minvar_lw", "meanvar", "maxsharpe_lw", "erc"]
    W = {k: np.zeros((T, n)) for k in names}
    shrink = []
    frontiers = []
    for t in range(M, T):
        win = ex[t - M:t]
        mu = win.mean(0)
        cov = np.cov(win, rowvar=False)
        cov_lw, delta = ledoit_wolf(win)
        shrink.append(delta)
        mu_shr = 0.5 * mu + 0.5 * mu.mean()  # grand-mean shrinkage (James-Stein flavoured, fixed 50%)
        W["1/N"][t] = np.ones(n) / n
        W["minvar"][t] = w_minvar(cov)
        W["minvar_lw"][t] = w_minvar(cov_lw)
        W["meanvar"][t] = w_maxsharpe(mu, cov)
        W["maxsharpe_lw"][t] = w_maxsharpe(mu_shr, cov_lw)
        W["erc"][t] = w_erc(cov_lw)
        if dates[t].month == 12:  # yearly frontier snapshot for the viewer (ex-ante, from the window)
            frontiers.append({
                "date": str(dates[t]), "pts": frontier(mu, cov),
                "assets": [[float(np.sqrt(cov[i, i] * 12)), float(mu[i] * 12)] for i in range(n)],
                "ports": {k: [float(np.sqrt(W[k][t] @ cov @ W[k][t] * 12)), float(mu @ W[k][t] * 12)] for k in names},
            })
    oos = slice(M, T)
    res = {"period": [str(dates[M]), str(dates[-1])], "months": T - M, "n_assets": n,
           "window": M, "lw_shrinkage_mean": float(np.mean(shrink)), "strategies": {}}
    series = {"dates": [str(d) for d in dates[oos]]}
    base = {}
    for c in COSTS_BP:
        for k in names:
            w = W[k][oos]
            rx = ex[oos]
            gross_ex = np.sum(w * rx, axis=1)
            # turnover: from drifted previous weights to new target weights
            rtot = R[oos]
            to = np.zeros(len(w))
            for i in range(1, len(w)):
                drift = w[i - 1] * (1 + rtot[i - 1])
                drift /= drift.sum()
                to[i] = np.abs(w[i] - drift).sum()
            net_ex = gross_ex - to * c / 1e4
            net_tot = net_ex + rf.values[oos]
            sr = net_ex.mean() / net_ex.std(ddof=1) * math.sqrt(12)
            mdd, dd = max_drawdown(net_tot)
            entry = {"ann_excess_return": float(net_ex.mean() * 12), "ann_vol": float(net_ex.std(ddof=1) * math.sqrt(12)),
                     "sharpe": float(sr), "ceq_gamma1": float((net_ex.mean() - 0.5 * net_ex.var(ddof=1)) * 12),
                     "turnover_monthly": float(to[1:].mean()), "max_drawdown": mdd}
            if k == "1/N":
                base[c] = net_ex
            res["strategies"].setdefault(k, {})[f"{c}bp"] = entry
            if c == COSTS_BP[0]:
                series[k] = {"wealth": np.cumprod(1 + net_tot).round(5).tolist(), "drawdown": dd.round(5).tolist(),
                             "weights_yearly": [W[k][M + i].round(4).tolist() for i in range(0, T - M, 12)]}
        for k in names:
            if k == "1/N":
                continue
            z, p = jk_memmel(_net(W, k, c, ex, R, oos), base[c])
            res["strategies"][k][f"{c}bp"]["jk_z_vs_1N"] = z
            res["strategies"][k][f"{c}bp"]["jk_p_vs_1N"] = p
    res["asset_names"] = list(ind.columns)
    res["wall_s"] = time.time() - t0
    (OUT / "backtest.json").write_text(json.dumps(res, indent=1))
    (OUT / "backtest_series.json").write_text(json.dumps({"series": series, "frontiers": frontiers}))
    print(json.dumps({k: v[f"{COSTS_BP[0]}bp"] for k, v in res["strategies"].items()}, indent=1))
    print(f"wall {res['wall_s']:.0f}s, LW delta mean {res['lw_shrinkage_mean']:.3f}")


def _net(W, k, c, ex, R, oos):
    """Net-of-cost monthly excess returns of strategy k (same turnover rule as in run())."""
    w = W[k][oos]; rx = ex[oos]; rtot = R[oos]
    gross = np.sum(w * rx, axis=1)
    to = np.zeros(len(w))
    for i in range(1, len(w)):
        d = w[i - 1] * (1 + rtot[i - 1]); d /= d.sum(); to[i] = np.abs(w[i] - d).sum()
    return gross - to * c / 1e4


if __name__ == "__main__":
    run()
