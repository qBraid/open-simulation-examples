# Finance and risk lab

Price derivatives, measure market risk on a GPU, and build portfolios
out of sample with open-source tools on qBraid. Every number is first checked
against a closed form or a published reference. The results render as a
three.js lab: a 15-year implied-volatility surface you can play like a film, a
Monte Carlo path fan with the VaR/ES tail lit up, and the efficient frontier
drifting through 52 years.

It stands in for the analytics layer of Bloomberg PORT, MSCI Barra/RiskMetrics
and Numerix. **The honest caveat comes first:** the maths here is at parity.
The commercial moat is data, meaning option chains, risk-model factor returns
and fundamentals. This example uses only public data (Ken French Data Library,
CBOE index history), which shapes what it can claim.

## What runs

| Piece | Tool | Role |
|---|---|---|
| Pricing | [QuantLib](https://www.quantlib.org) 1.43 (BSD) | Black-Scholes, Heston (Fourier, COS, FD), American and Bermudan (FD, lattice, approximations) |
| Market risk | [CuPy](https://cupy.dev) 14.2 on an NVIDIA L4 | Monte Carlo VaR/ES, full revaluation of a 500-option book |
| Portfolios | [cvxpy](https://www.cvxpy.org) 1.9 + Clarabel | Long-only min-variance and max-Sharpe, Ledoit-Wolf shrinkage, risk parity |
| Data | Ken French 30 industry portfolios, CBOE VIX9D/VIX/VIX3M/VIX6M/VIX1Y/SKEW | fetched by `fetch_data.sh`, not committed |
| Viewer | three.js 0.170 | vol surface history, path fan, frontier through time, backtest and validation pages |

## Reproduce

```bash
python3.12 -m venv venv && . venv/bin/activate
pip install QuantLib numpy scipy pandas cvxpy
pip install "cupy-cuda12x[ctk]" nvidia-cuda-cccl-cu12   # GPU only (mc_risk.py); [ctk] brings the CUDA runtime libs
./fetch_data.sh
python pricing.py        # ~3 min, 1 core   -> results/pricing.json
python backtest.py       # ~25 s, 1 core    -> results/backtest*.json
python vol_surface.py    # seconds          -> results/vol_surfaces.json
python mc_risk.py        # GPU, minutes     -> results/mc_risk.json
python build_viewer.py   # -> viewer.html
qbraid-canvas viewer.html --title "Finance and risk"   # or open it in a browser
```

Deep links: `viewer.html#vol@2020-03-16`, `#risk`, `#frontier@36`, `#backtest`, `#valid`.

## The bar: what "top 10%" means here

Finance has no leaderboard for "the best pricer". The professional standard is
three-fold:
- pricing engines agree with closed forms and published references to **basis-point** accuracy;
- Monte Carlo risk numbers carry **error bars that match theory**;
- backtests follow an **academic out-of-sample protocol**: no look-ahead, costs charged, significance tested against 1/N.

Each part has an objective test below. The verdicts are **reached** for pricing
and MC, and **reached (protocol), with a negative result** for portfolios.

### 1. Pricing: reached

| Test | Reference | Result |
|---|---|---|
| Black-Scholes, 36 European options | own closed form | analytic 1.5e-9 bp; FD 400x800 at most 3.3 bp; MC 400k within 3 SE in 36/36 |
| Heston, 10 strikes | Lewis (2000) reference prices | analytic and COS at most 7e-11 bp; FD 200x200x100 at most 1.95 bp |
| Heston | Fang & Oosterlee (2008): 5.785155450 | 2.7e-5 bp (the reference's 9-decimal rounding) |
| American puts, 20 cases | 20,001-step Leisen-Reimer lattice | FD 800x1600 at most 2.5 bp; Barone-Adesi-Whaley up to 235 bp; Bjerksund-Stensland up to 249 bp |
| Bermudan-50 puts, 20 cases | Longstaff & Schwartz (2001), Table 1 | 15/20 to the published 3 decimals; 5 high-vol 2-year cells off by at most 0.0058 (1.45 bp of strike) |

**A finding worth knowing.** The Longstaff & Schwartz Table 1 "finite
difference" column is widely quoted as American put prices. It is in fact the
value of an option exercisable **50 times a year**, which matches their LSM
grid. A continuous-exercise lattice is about 0.008 higher in every row (S=36,
sigma=0.2, T=1: 4.4867 vs 4.478). A Bermudan-50 FD price reproduces the table:
4.4778, 4.8402, 1.1099. In the 5 cells that still differ (sigma=0.4, T=2), our
FD grid is converged to 1e-5 (500 to 4000 nodes: 5.64117, 5.64121, 5.64122,
5.64122), so the residual is in the published values.

### 2. GPU Monte Carlo risk: see results below

MC_RESULTS_PLACEHOLDER

### 3. Portfolios: protocol reached, and 1/N still wins

The setup follows DeMiguel, Garlappi & Uppal (2009):
- 30 US industry portfolios (value-weighted, Ken French);
- out of sample July 1973 to August 2026, 638 months;
- a 120-month rolling window and monthly rebalancing;
- excess returns over the T-bill;
- costs on actual turnover, including drift.

| Strategy | Sharpe @10bp | Sharpe @50bp | p vs 1/N (10bp) | Turnover/mo | Max DD |
|---|---|---|---|---|---|
| 1/N | 0.510 | 0.502 | – | 3.0% | -53% |
| Min-variance (sample) | 0.540 | 0.512 | 0.74 | 7.2% | -38% |
| Min-variance (Ledoit-Wolf) | 0.559 | 0.533 | 0.56 | 6.6% | -39% |
| Mean-variance (sample) | 0.470 | 0.419 | 0.66 | 20.5% | -60% |
| Max-Sharpe (shrunk) | 0.513 | 0.469 | 0.97 | 15.5% | -49% |
| Risk parity (ERC) | 0.533 | 0.523 | 0.055 | 3.0% | -50% |

p is the Jobson-Korkie test with the Memmel (2003) correction. **No strategy
beats 1/N at 5% significance**, reproducing DGU's central result
with data through 2026. Sample mean-variance is the worst: estimation error in expected
returns swamps the optimisation gain, and turnover costs make it worse.
Minimum-variance strategies cut drawdowns from about 53% to 38-39% at a
similar Sharpe ratio. Risk parity comes closest to significance (p = 0.055).
That is the honest headline. If a user's backtest beats 1/N by a mile, look
for look-ahead bias first.

## Quantum readiness

- **Today:** portfolio selection as a QUBO and toy amplitude-estimation
  circuits run on simulators and small QPUs. They are research and teaching
  artefacts; the classical MC and convex solvers above win outright.
- **Future:** quantum amplitude estimation improves MC error from 1/sqrt(N) to
  1/N. Chakrabarti et al. (2021, *Quantum* 5, 463, "A threshold for quantum
  advantage in derivative pricing") estimate about **7.5k logical qubits** and a
  **T-depth of about 5.4e7** to price a benchmark autocallable and TARF at useful
  accuracy within a second. That is fault-tolerant hardware that does not exist
  yet. The practical step now is to get the classical pipeline (payoffs, models,
  validation) right, because that is the part that transfers.

## Implied-volatility surfaces: what they are and are not

Free historical S&P 500 option chains do not exist, so the vol tab shows a
**model surface anchored to public CBOE indices**:
- VIX9D, VIX, VIX3M, VIX6M and VIX1Y set the at-the-money term structure, interpolated in total variance;
- CBOE SKEW sets the slope and curvature of a bounded quadratic smile in standardised moneyness.

The smile coefficients are a heuristic sized to typical S&P 500 shapes, with a
30-day 90% strike at about 1.5x ATM vol. A Gram-Charlier expansion was tried
first and rejected: at SKEW-level skewness (about -4.5) it put 30-day put wings
above 150% vol. The surface shows real history (the VIX term structure inverts
in 38 of 196 months, and the 2020-03-16 peak is VIX 82.7) through a modelled
smile. With licensed option data the same viewer takes calibrated SVI slices
directly.

## Honest limits

- Public data only. No option chains, so no calibration to traded surfaces. The vol surface is an anchored model, as labelled.
- The backtest universe is 30 industry portfolios, not single stocks, and has no shorting, leverage or factor models (Barra-style). The covariance shrinkage target is scaled identity.
- The MC option book is synthetic (500 Black-Scholes options on 20 correlated GBM underlyings). Revaluation is exact Black-Scholes, not a vendor pricer.
- GPU speed-ups are measured against 2 CPU threads, the per-worker allowance on the shared pool box, not a full server socket.

## Verification stamp

STAMP_PLACEHOLDER
