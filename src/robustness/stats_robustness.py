"""Statistical robustness of the barrier fits (action plan S1, S3, S4). Stored data only.

S1  slope with and without the b log N term, corr(b, dV) from the fit covariance, and the
    fit after dropping the one or two smallest sizes; for the threshold times of campaign 1
    (data/crossover_exit_times.json) and the threshold and off-target times of campaign 2
    (data/collapse_time/raw).
S3  lattice rounding: add the regressor frac(N x*) = N x* - floor(N x*) to the paper's model;
    as a null, replace x* by x' drawn uniformly from (0.40, 0.72) and record how often the
    chi2 drop is at least as large as for x*.  For R_off, residuals at the sizes where
    N a_off or N b_off is not an integer.
S4  exponentiality of the off-target times per (r, N): sd/mean and median/mean, and
    sqrt(n) KS against the exponential with the censored-MLE mean truncated at the horizon.
    Campaign 1 stores only per-size summaries, so S4 uses campaign 2.

Writes data/robustness/stats.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "collapse_time"))
from common import (PAPER, RAW, RECT_LARGE, censored_mle, fit_slope, fit_slope_linear,  # noqa: E402
                    fixed_points, old_threshold_points, saddle_actions)

OUT = PAPER / "data" / "robustness"
RATES = (0.5, 1.0, 2.0, 4.0, 8.0)
N_NULL = 2000


def wls(N, E, k, extra=()):
    """WLS of log E on (N, log N, 1, *extra) with the paper's weights and se inflation."""
    N = np.asarray(N, float)
    y = np.log(np.asarray(E, float))
    w = np.asarray(k, float)
    A = np.vstack([N, np.log(N), np.ones_like(N), *extra]).T
    rt = np.sqrt(w)[:, None]
    sol, *_ = np.linalg.lstsq(rt * A, rt[:, 0] * y, rcond=None)
    cov = np.linalg.inv(A.T @ (w[:, None] * A))
    dof = len(N) - A.shape[1]
    resid = y - A @ sol
    red = float(np.sum(w * resid ** 2)) / dof if dof > 0 else float("nan")
    infl = max(1.0, red) if dof > 0 else float("nan")
    return dict(coef=sol.tolist(), se=np.sqrt(np.diag(cov) * infl).tolist(),
                corr_slope_logN=float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])),
                chi2=float(np.sum(w * resid ** 2)), chi2_red=red, dof=dof,
                resid=resid.tolist())


def s1(N, E, k):
    N, E, k = np.asarray(N), np.asarray(E), np.asarray(k)
    out = dict(sizes=N.tolist(), with_logN=fit_slope(N, E, k), without_logN=fit_slope_linear(N, E, k),
               corr_slope_logN=wls(N, E, k)["corr_slope_logN"])
    for m in (1, 2):
        if len(N) - m >= 4:
            out[f"drop_{m}_smallest"] = dict(with_logN=fit_slope(N[m:], E[m:], k[m:]),
                                             without_logN=fit_slope_linear(N[m:], E[m:], k[m:]))
    return out


def s3(N, E, k, x_star, rng):
    N = np.asarray(N, float)
    if len(N) < 6:
        return dict(note=f"only {len(N)} sizes; four-parameter fit not attempted")
    frac = lambda x: N * x - np.floor(N * x + 1e-9)
    base = wls(N, E, k)
    fx = wls(N, E, k, extra=(frac(x_star),))
    drop = base["chi2"] - fx["chi2"]
    null = np.array([base["chi2"] - wls(N, E, k, extra=(frac(x),))["chi2"]
                     for x in rng.uniform(0.40, 0.72, N_NULL)])
    return dict(frac_values=frac(x_star).tolist(),
                baseline=dict(slope=base["coef"][0], se=base["se"][0], chi2_red=base["chi2_red"], dof=base["dof"]),
                with_frac=dict(slope=fx["coef"][0], se=fx["se"][0], frac_coef=fx["coef"][3],
                               frac_se=fx["se"][3], chi2_red=fx["chi2_red"], dof=fx["dof"]),
                chi2_drop=drop, null_x_range=[0.40, 0.72], null_draws=N_NULL,
                null_p_value=float((1 + np.sum(null >= drop)) / (1 + N_NULL)),
                null_drop_quantiles=dict(zip(["q50", "q90", "q99"], np.quantile(null, [.5, .9, .99]).tolist())))


def s4(t, H):
    t = np.asarray(t, float)
    m, k = censored_mle(t, np.isfinite(t), H)
    n = len(t)
    obs = t[np.isfinite(t) & (t <= H)]
    cdf = lambda s: (1 - np.exp(-s / m)) / (1 - np.exp(-H / m))
    ks = st.kstest(obs, cdf)
    return dict(n=n, n_events=k, censored_frac=1 - k / n, mle_mean=m,
                median_over_mle_mean=float(np.median(t) / m) if k > n / 2 else None,
                sd_over_mean_observed=float(obs.std(ddof=1) / obs.mean()),
                ks_stat=float(ks.statistic), sqrtn_ks=float(np.sqrt(k) * ks.statistic), ks_p=float(ks.pvalue))


def load_campaign2(r):
    pts = []
    for f in sorted(RAW.glob(f"r{r:g}_N*.npz")):
        meta = json.loads(f.with_suffix(".json").read_text())
        pts.append((meta, dict(np.load(f))))
    pts.sort(key=lambda p: p[0]["N"])
    return pts


def main():
    rng = np.random.default_rng(20260929)
    x_star, _ = fixed_points()
    dV = saddle_actions()
    old = old_threshold_points()
    res = dict(settings=dict(x_star=x_star, rect_off=RECT_LARGE, null_draws=N_NULL, seed=20260929,
                             model="WLS log E = a + b log N + N dV [+ g frac(N x*)], weights n_exit, "
                                   "se inflated by max(1, chi2_red)^0.5"),
               rates={})
    for r in RATES:
        R = dict(saddle_dV_num=dV[r])
        c1 = old[f"{r:.1f}"]
        N1, E1, k1 = ([q[f] for q in c1] for f in ("N", "Etau", "n_exit"))
        R["campaign1_threshold"] = dict(S1=s1(N1, E1, k1), S3=s3(N1, E1, k1, x_star, rng))
        pts = load_campaign2(r)
        if pts:
            N2 = np.array([m["N"] for m, _ in pts])
            for e, key in (("thr", "tau_thr"), ("off", "tau_off")):
                est = [censored_mle(d[key], np.isfinite(d[key]), m["horizon"]) for m, d in pts]
                E2, k2 = [a for a, _ in est], [b for _, b in est]
                block = dict(S1=s1(N2, E2, k2))
                if e == "thr":
                    block["S3"] = s3(N2, E2, k2, x_star, rng)
                else:
                    base = wls(N2, E2, k2)
                    z = np.array(base["resid"]) * np.sqrt(k2)
                    block["S3_rect"] = dict(
                        nonintegral_sizes=[int(N) for N in N2
                                           if any(abs(N * a - round(N * a)) > 1e-9 for a in RECT_LARGE)],
                        standardized_resid=dict(zip(map(int, N2), z.tolist())),
                        chi2_red=base["chi2_red"], dof=base["dof"])
                    block["S4"] = {int(m["N"]): s4(d[key], m["horizon"]) for m, d in pts}
                R[f"campaign2_{e}"] = block
        res["rates"][f"{r:g}"] = R

        c = R["campaign1_threshold"]["S1"]
        print(f"r={r:g}: dV_num {dV[r]:.4f} | c1 thr {c['with_logN']['slope']:.4f} / lin "
              f"{c['without_logN']['slope']:.4f}, corr {c['corr_slope_logN']:+.3f}")
        for tag, s in (("c1", R["campaign1_threshold"]["S3"]), ("c2", R.get("campaign2_thr", {}).get("S3", {}))):
            if "with_frac" in s:
                print(f"   S3 {tag}: chi2r {s['baseline']['chi2_red']:.2f} -> {s['with_frac']['chi2_red']:.2f}, "
                      f"g {s['with_frac']['frac_coef']:.3f}+-{s['with_frac']['frac_se']:.3f}, "
                      f"null p {s['null_p_value']:.3f}")
        for e in ("thr", "off"):
            if f"campaign2_{e}" in R:
                c = R[f"campaign2_{e}"]["S1"]
                print(f"   c2 {e}: {c['with_logN']['slope']:.4f} (chi2r {c['with_logN']['chi2_red']:.2f}) / lin "
                      f"{c['without_logN']['slope']:.4f}, corr {c['corr_slope_logN']:+.3f}")
        if "campaign2_off" in R:
            q = list(R["campaign2_off"]["S4"].values())
            print("   S4 off: sd/mean {:.3f}-{:.3f}, sqrt(n)KS {:.2f}-{:.2f}, nonintegral N {}".format(
                min(v["sd_over_mean_observed"] for v in q), max(v["sd_over_mean_observed"] for v in q),
                min(v["sqrtn_ks"] for v in q), max(v["sqrtn_ks"] for v in q),
                R["campaign2_off"]["S3_rect"]["nonintegral_sizes"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stats.json").write_text(json.dumps(res, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
