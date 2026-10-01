---
name: quant-finance
description: Price derivatives, measure market risk (VaR/ES) and build portfolios on qBraid with open-source tools (QuantLib, CuPy/NumPy Monte Carlo on GPU, cvxpy/Riskfolio-Lib, Ken French and CBOE public data). Use when a user brings an option-pricing, risk, backtesting or portfolio-construction problem, or asks for an open alternative to Bloomberg PORT, MSCI Barra/RiskMetrics or Numerix. Gives the validate-first recipes, the benchmark references, the out-of-sample protocol, GPU sizing and where quantum (amplitude estimation, portfolio QUBO) honestly stands.
metadata:
  version: "0.1.0"
  layer: "1"
  status: "draft"
  verified: "2026-10-01"
---

# Quant finance stack on qBraid

The maths is open and at parity. The commercial moat is **data** (option
chains, fundamentals, risk-model factor returns), not solvers. Say so up front,
then use public data or the user's own feeds.

## Pick the tool (decision rules)

| Task | First choice | Notes |
|---|---|---|
| Vanilla and exotic pricing, curves, calendars | QuantLib (`pip install QuantLib`, 1.43) | Analytic, FD, lattice and MC engines. Validate each engine against a closed form or a published table before trusting it. |
| Heston / stochastic vol | QuantLib `AnalyticHestonEngine` or `COSHestonEngine` | Matches Lewis (2000) reference prices to ~1e-12 bp. Use FD only when you need early exercise or barriers. |
| American options | QuantLib FD (`FdBlackScholesVanillaEngine`, 800x1600) | Within 2.6 bp of a 20,001-step Leisen-Reimer lattice. Barone-Adesi-Whaley and Bjerksund-Stensland approximations can be off by up to ~250 bp; never use them for anything you report. |
| Portfolio VaR/ES, full revaluation | CuPy on GPU (`cupy-cuda12x[ctk]`) | An L4 revalues a 500-option book at 0.72M scenarios/s in FP64, 15x NumPy on 2 threads. A linear book has a closed form; use it as the known answer. |
| Portfolio optimisation | cvxpy + Clarabel (long-only QPs), Riskfolio-Lib for risk measures | Shrink the covariance (Ledoit-Wolf). Never trust sample-mean optimisation out of sample. |
| Public data | Ken French Data Library (returns, factors), CBOE index history (VIX family, SKEW), FRED | Fetch at run time; do not commit raw files (publisher terms). |

## Validate first (the known answers)

Run these before any user-facing number. The results below were verified on 2026-10-01.

1. **Black-Scholes:** QuantLib analytic against your own closed form. Expect about 1e-9 bp. FD 400x800 should be at most 5 bp. MC must be within 3 SE in every case (36/36).
2. **Heston:** Lewis (2000) prices (S=100, r=1%, q=2%, v0=0.04, kappa=4, theta=0.25, sigma=1, rho=-0.5, T=1). Calls K=80..120 are 26.774758743998854, 20.933349000596710, 16.070154917028834, 12.132211516709845, 9.024913483457836. Fang & Oosterlee (2008): 5.785155450. Analytic and COS reproduce all of them to at most 3e-5 bp.
3. **American puts:** the trap. The Longstaff & Schwartz (2001) Table 1 "finite difference" values (S=36, sigma=0.2, T=1 gives 4.478) are for **50 exercise dates a year (Bermudan)**, not continuous exercise. A continuous-exercise lattice gives 4.4867, about 0.008 higher in every row. Price a Bermudan-50 option to compare. It matches 15/20 cells to the published 3 decimals. The 5 high-vol, 2-year cells differ by at most 0.0058, where our grid is converged to 1e-5.
4. **MC risk:** on a jointly normal linear book, MC VaR/ES must converge to the closed form, and the spread over independent replications must match the asymptotic SE: SE(VaR) = sqrt(a(1-a)/N)/f(VaR), and SE(ES) = sqrt((Var(L|L>=VaR) + a(ES-VaR)^2)/((1-a)N)). A ratio of SD to theory near 1 means the error bars are right.

## Backtests: the protocol that makes them believable

Follow DeMiguel, Garlappi & Uppal (2009):
- Use a rolling estimation window (120 months) and monthly rebalancing.
- Weights at t use data up to t-1 only.
- Use excess returns over the T-bill.
- Charge costs on the actual turnover, including weight drift.
- Report Sharpe, CEQ, turnover and max drawdown, plus the Jobson-Korkie/Memmel p-value against 1/N.

**Expect 1/N to be hard to beat.** On 30 US industries, 1973-2026, at 10 bp:
- Ledoit-Wolf min-variance has Sharpe 0.559 against 0.510 for 1/N (p = 0.56).
- Sample mean-variance is the worst, at 0.470.
- Nothing is significant at 5%.

If a user's backtest shows a big win, check for look-ahead and missing costs first.

## Compute on qBraid

- **Install CuPy with its toolkit extra:** `pip install "cupy-cuda12x[ctk]" nvidia-cuda-cccl-cu12`. Plain `cupy-cuda12x` imports fine, then fails at the first random draw with `Failure finding libcurand.so`. qBraid GPU images ship the driver but no CUDA toolkit.

- **Pricing and backtests:** single-threaded and seconds to minutes. The pod is fine.
- **GPU MC:** `gpu-l4` ($0.49/h) is enough. The full validation (32 replications up to 1e7, 8 at 1e8) plus a 2M-scenario option-book revaluation and the path fan took 32 s on an L4. Expect the GPU queue, not the compute, to dominate on a shared box.
- FP64 is slow on the L4. If you need 1e9+ scenarios in double precision, use an A100 or H100, or validate FP32 against FP64 first.
- Shared pool boxes: one GPU job at a time, and ~5 real CPUs regardless of `nproc`.

## Quantum: where it honestly stands

- **Today:** portfolio selection as a QUBO and toy amplitude-estimation circuits run on simulators and small QPUs. They are research and teaching artefacts. Classical MC and convex solvers win outright.
- **Future:** amplitude estimation gives a quadratic speedup in MC error, 1/N instead of 1/sqrt(N). Chakrabarti et al. (2021, *Quantum* 5, 463) estimate ~7.5k logical qubits and a T-depth of ~5.4e7 for a benchmark derivative priced to useful accuracy within one second. That is fault-tolerant hardware that does not exist yet. Position this as "know your workload now; the pipeline transfers later".

## Recipe (this repo)

`finance-risk/` in qBraid/open-simulation-examples:

```
./fetch_data.sh                          # Ken French + CBOE (public)
python pricing.py                         # ~3 min, 1 core
python backtest.py                        # ~25 s, 1 core
python vol_surface.py                     # seconds
python mc_risk.py                         # GPU (cupy-cuda12x), minutes on an L4
python build_viewer.py                    # -> viewer.html, then: qbraid-canvas viewer.html
```
