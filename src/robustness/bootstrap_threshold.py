"""Parametric bootstrap of the barrier slopes (action plan S2).

Campaign 1 (data/crossover_exit_times.json) stores only E[tau], n_exit, n_rep and the horizon per
size, so the trajectory-level bootstrap of analyze.py cannot be run on it.  Here each size is
resampled as n_rep exponential times with the fitted mean, right-censored at the stored horizon,
and re-estimated with the same censored MLE; the fits are then redone.  This measures sampling
noise under exponentiality only: it checks the nominal errors, not the chi2-inflated ones.

Validation: the same parametric bootstrap on campaign 2, compared with the trajectory-level
bootstrap stored in data/collapse_time/fit_results.json.

Writes data/robustness/bootstrap.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "collapse_time"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXP, PAPER, censored_mle, fit_slope, fit_slope_linear, fixed_points, old_threshold_points  # noqa: E402
from stats_robustness import RATES, load_campaign2, wls  # noqa: E402

OUT = PAPER / "data" / "robustness"
B = 2000


def resample(rng, E, n, H):
    t = rng.exponential(E, n)
    t[t > H] = np.inf
    return censored_mle(t, np.isfinite(t), H)


def bootstrap(rng, N, E, nrep, H, x_star):
    N = np.asarray(N, float)
    frac = N * x_star - np.floor(N * x_star + 1e-9)
    with_frac = len(N) >= 6
    s = {"with_logN": [], "without_logN": []} | ({"with_frac": []} if with_frac else {})
    for _ in range(B):
        est = [resample(rng, e, n, h) for e, n, h in zip(E, nrep, H)]
        Eb, kb = [a for a, _ in est], [b for _, b in est]
        s["with_logN"].append(fit_slope(N, Eb, kb)["slope"])
        s["without_logN"].append(fit_slope_linear(N, Eb, kb)["slope"])
        if with_frac:
            s["with_frac"].append(wls(N, Eb, kb, extra=(frac,))["coef"][0])
    k = [min(n, max(1, round(n * (1 - np.exp(-h / e))))) for e, n, h in zip(E, nrep, H)]
    point = {"with_logN": fit_slope(N, E, k), "without_logN": fit_slope_linear(N, E, k)}
    out = {}
    for m, v in s.items():
        v = np.array(v)
        out[m] = dict(bootstrap_se=float(v.std(ddof=1)), ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                      bias=float(v.mean() - (point[m]["slope"] if m in point else v.mean())))
    out["with_logN"]["se_nominal"] = point["with_logN"]["se_nominal"]
    out["with_logN"]["se_inflated"] = point["with_logN"]["se"]
    out["with_logN"]["chi2_red"] = point["with_logN"]["chi2_red"]
    return out


def main():
    rng = np.random.default_rng(20260930)
    x_star, _ = fixed_points()
    old = old_threshold_points()
    fr = json.loads((EXP / "fit_results.json").read_text())["rates"]
    res = dict(settings=dict(B=B, seed=20260930, model="exponential with the fitted mean, censored at the stored horizon"),
               rates={})
    for r in RATES:
        c1 = old[f"{r:.1f}"]
        R = dict(campaign1_threshold=bootstrap(rng, [q["N"] for q in c1], [q["Etau"] for q in c1],
                                               [q["n_rep"] for q in c1], [q["horizon"] for q in c1], x_star))
        pts = load_campaign2(r)
        for e, key in (("thr", "tau_thr"), ("off", "tau_off")):
            est = [censored_mle(d[key], np.isfinite(d[key]), m["horizon"]) for m, d in pts]
            b = bootstrap(rng, [m["N"] for m, _ in pts], [a for a, _ in est], [len(d[key]) for _, d in pts],
                          [m["horizon"] for m, _ in pts], x_star)
            b["with_logN"]["trajectory_bootstrap_se"] = fr[f"{r:g}"]["fits"][e]["bootstrap_se"]
            R[f"campaign2_{e}"] = b
        res["rates"][f"{r:g}"] = R
        for name, b in R.items():
            w = b["with_logN"]
            print(f"r={r:g} {name:20s} se: param {w['bootstrap_se']:.4f}, nominal {w['se_nominal']:.4f}, "
                  f"inflated {w['se_inflated']:.4f} (chi2r {w['chi2_red']:.2f})"
                  + (f", trajectory {w['trajectory_bootstrap_se']:.4f}" if "trajectory_bootstrap_se" in w else "")
                  + f" | linear {b['without_logN']['bootstrap_se']:.4f}"
                  + (f" | frac {b['with_frac']['bootstrap_se']:.4f}" if "with_frac" in b else ""), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bootstrap.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
