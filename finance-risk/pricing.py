"""Pricing validation: QuantLib engines against closed forms and published references.

Three blocks, each scored in basis points (bp):
  1. Black-Scholes European options: QuantLib analytic, finite-difference and
     Monte Carlo engines against our own closed-form implementation.
  2. Heston: QuantLib analytic (Fourier), COS and finite-difference engines
     against published reference prices (Lewis 2000; Fang & Oosterlee 2008).
  3. American puts: QuantLib FD and analytic approximations against the
     Longstaff & Schwartz (2001, Table 1) finite-difference references and our
     own 20,000-step Leisen-Reimer lattice.

Errors are reported as |price - reference| / reference * 1e4 (relative bp) and,
where the reference is a published 3-decimal value, as absolute error in bp of
strike so rounding is visible.
"""
import json
import math
import time
from pathlib import Path

import QuantLib as ql
from scipy.stats import norm

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

TODAY = ql.Date(15, 1, 2025)
ql.Settings.instance().evaluationDate = TODAY
DC = ql.Actual365Fixed()
CAL = ql.NullCalendar()


def bp(x, ref):
    return abs(x - ref) / abs(ref) * 1e4


def flat(rate):
    return ql.YieldTermStructureHandle(ql.FlatForward(TODAY, rate, DC))


def maturity(t):
    return TODAY + ql.Period(int(round(t * 365)), ql.Days)


def bs_closed(s, k, t, r, q, vol, kind):
    d1 = (math.log(s / k) + (r - q + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    if kind == "call":
        return s * math.exp(-q * t) * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)
    return k * math.exp(-r * t) * norm.cdf(-d2) - s * math.exp(-q * t) * norm.cdf(-d1)


def bsm_process(s, r, q, vol):
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(s)), flat(q), flat(r),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(TODAY, CAL, vol, DC)))


def european(k, t, kind):
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if kind == "call" else ql.Option.Put, k)
    return ql.VanillaOption(payoff, ql.EuropeanExercise(maturity(t)))


def block_bs():
    rows = []
    cases = [(100, k, t, 0.03, 0.01, v, kind)
             for k in (80, 100, 120) for t in (0.25, 1.0, 2.0) for v in (0.15, 0.35)
             for kind in ("call", "put")]
    for s, k, t, r, q, vol, kind in cases:
        t_eff = DC.yearFraction(TODAY, maturity(t))
        ref = bs_closed(s, k, t_eff, r, q, vol, kind)
        opt = european(k, t, kind)
        proc = bsm_process(s, r, q, vol)
        res = {"S": s, "K": k, "T": t, "vol": vol, "type": kind, "closed_form": ref}
        opt.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
        res["analytic"] = opt.NPV()
        opt.setPricingEngine(ql.FdBlackScholesVanillaEngine(proc, 400, 800))
        res["fd_400x800"] = opt.NPV()
        opt.setPricingEngine(ql.MCEuropeanEngine(proc, "pseudorandom", timeSteps=1,
                                                 requiredSamples=400_000, seed=42))
        res["mc_400k"] = opt.NPV()
        res["mc_400k_stderr"] = opt.errorEstimate()
        for key in ("analytic", "fd_400x800", "mc_400k"):
            res[key + "_bp"] = bp(res[key], ref)
        res["mc_within_3se"] = abs(res["mc_400k"] - ref) <= 3 * res["mc_400k_stderr"]
        rows.append(res)
    return rows


# Lewis (2000), "Option Valuation under Stochastic Volatility", reference prices
# for S=100, r=0.01, q=0.02, v0=0.04, kappa=4, theta=0.25, sigma=1, rho=-0.5, T=1.
LEWIS = {
    "params": dict(s=100.0, r=0.01, q=0.02, v0=0.04, kappa=4.0, theta=0.25, sigma=1.0, rho=-0.5, t=1.0),
    "calls": {80: 26.774758743998854, 90: 20.933349000596710, 100: 16.070154917028834,
              110: 12.132211516709845, 120: 9.024913483457836},
    "puts": {80: 7.958878113256768, 90: 12.017966707346304, 100: 17.055270961270109,
             110: 23.017825898442800, 120: 29.811026202682471},
    "source": "Lewis, A. (2000). Option Valuation under Stochastic Volatility. Finance Press. "
              "Reference values as reproduced in the FFT/COS literature (e.g. Fang & Oosterlee 2008).",
}
# Fang & Oosterlee (2008), SIAM J. Sci. Comput. 31(2), Heston test: reference call 5.785155450.
FO = {
    "params": dict(s=100.0, r=0.0, q=0.0, v0=0.0175, kappa=1.5768, theta=0.0398, sigma=0.5751, rho=-0.5711, t=1.0),
    "calls": {100: 5.785155450},
    "source": "Fang, F. & Oosterlee, C.W. (2008). A novel pricing method for European options based on "
              "Fourier-cosine series expansions. SIAM J. Sci. Comput. 31(2), Heston test case.",
}


def heston_model(p):
    proc = ql.HestonProcess(flat(p["r"]), flat(p["q"]), ql.QuoteHandle(ql.SimpleQuote(p["s"])),
                            p["v0"], p["kappa"], p["theta"], p["sigma"], p["rho"])
    return ql.HestonModel(proc)


def block_heston():
    rows = []
    for name, ref in (("Lewis 2000", LEWIS), ("Fang & Oosterlee 2008", FO)):
        p = ref["params"]
        # Use an exact 365-day maturity so T = 1.0 exactly under Actual365Fixed.
        model = heston_model(p)
        engines = {
            "analytic_fourier": ql.AnalyticHestonEngine(model, 1e-12, 100_000),
            "cos": ql.COSHestonEngine(model, 25, 1024),
            "fd_200x100x100": ql.FdHestonVanillaEngine(model, 200, 200, 100),
        }
        for kind in ("calls", "puts"):
            for k, refp in ref.get(kind, {}).items():
                opt = european(k, p["t"], "call" if kind == "calls" else "put")
                res = {"set": name, "K": k, "type": kind[:-1], "reference": refp}
                for en, eng in engines.items():
                    opt.setPricingEngine(eng)
                    t0 = time.perf_counter()
                    v = opt.NPV()
                    res[en] = v
                    res[en + "_bp"] = bp(v, refp)
                    res[en + "_ms"] = (time.perf_counter() - t0) * 1e3
                rows.append(res)
    return rows, {"Lewis 2000": LEWIS["source"], "Fang & Oosterlee 2008": FO["source"]}


# Longstaff & Schwartz (2001), Rev. Fin. Stud. 14(1), Table 1, finite-difference
# column. K=40, r=0.06. Keys: (S, sigma, T). NOTE: these values are for an option
# exercisable 50 times per year (Bermudan), matching the LSM exercise grid, not for
# continuous exercise. Comparing them with a continuous-exercise lattice shows a
# systematic ~0.008 gap; we score the Bermudan-50 price against the table and the
# continuous-exercise engines against our own 20,001-step lattice.
LS_TABLE1 = {
    (36, 0.2, 1): 4.478, (36, 0.2, 2): 4.840, (36, 0.4, 1): 7.101, (36, 0.4, 2): 8.508,
    (38, 0.2, 1): 3.250, (38, 0.2, 2): 3.745, (38, 0.4, 1): 6.148, (38, 0.4, 2): 7.670,
    (40, 0.2, 1): 2.314, (40, 0.2, 2): 2.885, (40, 0.4, 1): 5.312, (40, 0.4, 2): 6.920,
    (42, 0.2, 1): 1.617, (42, 0.2, 2): 2.212, (42, 0.4, 1): 4.582, (42, 0.4, 2): 6.248,
    (44, 0.2, 1): 1.110, (44, 0.2, 2): 1.690, (44, 0.4, 1): 3.948, (44, 0.4, 2): 5.647,
}


def block_american():
    rows = []
    k, r = 40.0, 0.06
    for (s, vol, t), ref in LS_TABLE1.items():
        proc = bsm_process(s, r, 0.0, vol)
        payoff = ql.PlainVanillaPayoff(ql.Option.Put, k)
        opt = ql.VanillaOption(payoff, ql.AmericanExercise(TODAY, maturity(t)))
        res = {"S": s, "vol": vol, "T": t, "LS2001_fd": ref}
        t0 = time.perf_counter()
        opt.setPricingEngine(ql.BinomialVanillaEngine(proc, "lr", 20001))
        res["lattice_LR_20001"] = opt.NPV()
        res["lattice_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        opt.setPricingEngine(ql.FdBlackScholesVanillaEngine(proc, 800, 1600))
        res["fd_800x1600"] = opt.NPV()
        res["fd_ms"] = (time.perf_counter() - t0) * 1e3
        opt.setPricingEngine(ql.BaroneAdesiWhaleyApproximationEngine(proc))
        res["baw"] = opt.NPV()
        opt.setPricingEngine(ql.BjerksundStenslandApproximationEngine(proc))
        res["bjerksund_stensland"] = opt.NPV()
        dates = [TODAY + int(round(365 * t * i / (50 * t))) for i in range(1, int(50 * t) + 1)]
        berm = ql.VanillaOption(payoff, ql.BermudanExercise(dates))
        berm.setPricingEngine(ql.FdBlackScholesVanillaEngine(proc, 2000, 2000))
        res["bermudan50_fd_2000x2000"] = berm.NPV()
        res["bermudan50_vs_LS2001_abs"] = abs(res["bermudan50_fd_2000x2000"] - ref)
        res["bermudan50_vs_LS2001_bp_of_strike"] = res["bermudan50_vs_LS2001_abs"] / k * 1e4
        res["bermudan50_within_rounding"] = res["bermudan50_vs_LS2001_abs"] <= 0.0005 + 1e-9
        lat = res["lattice_LR_20001"]
        for key in ("fd_800x1600", "baw", "bjerksund_stensland"):
            res[key + "_bp_vs_lattice"] = bp(res[key], lat)
        # Published values are rounded to 1e-3: compare in bp of strike so the
        # rounding floor (0.0005/40 = 0.125 bp of strike) is explicit.
        res["lattice_vs_LS2001_bp_of_strike"] = abs(lat - ref) / k * 1e4
        res["lattice_vs_LS2001_within_rounding"] = abs(lat - ref) <= 0.0005 + 1e-9
        rows.append(res)
    return rows


def summary(bs, hes, am):
    def mx(rows, key):
        return max(r[key] for r in rows)
    return {
        "bs_analytic_max_bp": mx(bs, "analytic_bp"),
        "bs_fd_max_bp": mx(bs, "fd_400x800_bp"),
        "bs_mc_within_3se": sum(r["mc_within_3se"] for r in bs),
        "bs_cases": len(bs),
        "heston_analytic_max_bp": mx(hes, "analytic_fourier_bp"),
        "heston_cos_max_bp": mx(hes, "cos_bp"),
        "heston_fd_max_bp": mx(hes, "fd_200x100x100_bp"),
        "heston_cases": len(hes),
        "american_lattice_vs_LS2001_max_bp_of_strike": mx(am, "lattice_vs_LS2001_bp_of_strike"),
        "american_lattice_within_rounding": sum(r["lattice_vs_LS2001_within_rounding"] for r in am),
        "bermudan50_vs_LS2001_max_bp_of_strike": mx(am, "bermudan50_vs_LS2001_bp_of_strike"),
        "bermudan50_within_rounding": sum(r["bermudan50_within_rounding"] for r in am),
        "american_fd_max_bp_vs_lattice": mx(am, "fd_800x1600_bp_vs_lattice"),
        "american_baw_max_bp_vs_lattice": mx(am, "baw_bp_vs_lattice"),
        "american_bjs_max_bp_vs_lattice": mx(am, "bjerksund_stensland_bp_vs_lattice"),
        "american_cases": len(am),
    }


def main():
    t0 = time.time()
    bs = block_bs()
    hes, sources = block_heston()
    am = block_american()
    out = {"quantlib": ql.__version__, "black_scholes": bs, "heston": hes,
           "heston_sources": sources, "american": am,
           "american_source": "Longstaff, F.A. & Schwartz, E.S. (2001). Valuing American Options by "
                              "Simulation: A Simple Least-Squares Approach. Rev. Financ. Stud. 14(1), "
                              "Table 1 (finite-difference column).",
           "summary": summary(bs, hes, am), "wall_s": time.time() - t0}
    (OUT / "pricing.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item()))
    print(json.dumps(out["summary"], indent=1, default=lambda o: o.item()), f"\nwall {out['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
