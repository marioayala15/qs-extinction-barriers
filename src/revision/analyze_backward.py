"""Analysis of the backward-equation exit times (referee point 3).

Reads data/revision/backward_equation/results.json and the stored simulations, and writes
data/revision/backward_equation/analysis.json:

  1. prefactor  E[tau] exp(-N dV_num)  and local slopes of log E[tau];
  2. fits  log E[tau] = a + N dV + c/N (+ e/N^2)  to the backward-equation values, for
     several smallest sizes, to measure the model error of the extrapolation;
  3. standardized differences between the simulated means and the backward equation;
  4. refits of the simulated means: a + N dV, a + b log N + N dV, a + N dV + c/N, with the
     smallest sizes dropped, and normal and Student-t p-values against dV_num and dV_inf.

    python src/revision/analyze_backward.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BKE = ROOT / "data" / "revision" / "backward_equation"
X_STAR = 0.5588112173244356
DV_INF = 0.061379490579627626
DV_THR = {0.5: 0.11308}          # computed threshold barrier where it is below the saddle


def wls(cols, y, w=None):
    A = np.vstack(cols).T
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    rt = np.sqrt(w)
    sol, *_ = np.linalg.lstsq(rt[:, None] * A, rt * y, rcond=None)
    cov = np.linalg.inv(A.T @ (w[:, None] * A))
    dof = len(y) - A.shape[1]
    chi2 = float(np.sum(w * (y - A @ sol) ** 2))
    return sol, cov, dof, chi2


def model_cols(N, model):
    N = np.asarray(N, float)
    cols = [N, np.ones_like(N)]
    if model == "logN":
        cols.append(np.log(N))
    if model in ("invN", "invN2"):
        cols.append(1.0 / N)
    if model == "invN2":
        cols.append(1.0 / N**2)
    return cols


def main():
    res = json.loads((BKE / "results.json").read_text())
    ssa = json.loads((ROOT / "data" / "collapse_time" / "fit_results.json").read_text())["rates"]
    camp1 = json.loads((ROOT / "data" / "crossover_exit_times.json").read_text())
    out = {}
    zs = {"off": [], "thr": [], "small": [], "thr_campaign1": []}
    for key, S in ssa.items():
        r = float(key)
        dV = S["saddle_dV_num"]
        tab = {w: {x["N"]: x for x in res["rows"] if x["r"] == r and x["target"] == w}
               for w in ("off", "thr", "small")}
        Ns = np.array(sorted(tab["off"]))
        logt = np.array([np.log(tab["off"][N]["tau"]) for N in Ns])
        R = dict(dV_num=dV, N=Ns.tolist(), tau_off=np.exp(logt).tolist(),
                 prefactor=np.exp(logt - Ns * dV).tolist())

        # direct solver where available
        R["direct_over_renewal_minus_1"] = {int(N): tab["off"][N]["tau_direct"] / tab["off"][N]["tau"] - 1
                                            for N in Ns if "tau_direct" in tab["off"][N]}

        # local slopes on the grid of step 20 (and step 10 below 160)
        grid = [N for N in Ns if N % 20 == 0]
        R["local_slopes"] = [dict(N_lo=int(a), N_hi=int(b),
                                  slope=float((np.log(tab["off"][b]["tau"]) - np.log(tab["off"][a]["tau"])) / (b - a)))
                             for a, b in zip(grid[:-1], grid[1:])]

        # extrapolation fits to the backward-equation values
        fits = []
        for model in ("lin", "invN", "invN2"):
            for nmin in (40, 80, 120, 160, 200, 300):
                m = Ns >= nmin
                if m.sum() < len(model_cols(Ns[m], model)) + 2:
                    continue
                sol, cov, dof, chi2 = wls(model_cols(Ns[m], model), logt[m])
                fits.append(dict(model=model, N_min=nmin, n=int(m.sum()), dV=float(sol[0]),
                                 dV_minus_num=float(sol[0] - dV), prefactor_limit=float(np.exp(sol[1])),
                                 c=float(sol[2]) if model != "lin" else None,
                                 rms_resid=float(np.sqrt(chi2 / max(dof, 1)))))
        R["bke_fits"] = fits

        # threshold times from the backward equation
        Nt = np.array(sorted(tab["thr"]))
        lt = np.array([np.log(tab["thr"][N]["tau"]) for N in Nt])
        frac = Nt * X_STAR - np.floor(Nt * X_STAR)
        tf = []
        for nmin in (40, 120, 200):
            m = Nt >= nmin
            sol, cov, dof, chi2 = wls([Nt[m], np.ones(m.sum()), 1.0 / Nt[m], frac[m]], lt[m])
            tf.append(dict(N_min=nmin, dV_thr=float(sol[0]), c=float(sol[2]), g=float(sol[3]),
                           rms_resid=float(np.sqrt(chi2 / max(dof, 1)))))
        R["thr_fits_with_lattice_term"] = tf
        R["thr_reference"] = DV_THR.get(r, dV)
        R["small_over_off_minus_1"] = {int(N): tab["small"][N]["tau"] / tab["off"][N]["tau"] - 1 for N in Ns
                                       if N in tab["small"]}

        # simulations against the backward equation
        pts = S["points"]
        zrow = []
        for p in pts:
            N = p["N"]
            for w in ("off", "thr", "small"):
                if w in p and p[w].get("n_exit", 0) > 0 and N in tab[w]:
                    z = (np.log(p[w]["Etau"]) - np.log(tab[w][N]["tau"])) / p[w]["se_log"]
                    zs[w].append(z)
                    if w == "off":
                        zrow.append(dict(N=N, z=float(z)))
        R["ssa_off_z"] = zrow
        for p in camp1.get(f"{r:.1f}", []):
            if p["N"] in tab["thr"]:
                zs["thr_campaign1"].append((np.log(p["Etau"]) - np.log(tab["thr"][p["N"]]["tau"])) * np.sqrt(p["n_exit"]))

        # refits of the simulated extinction times
        Np = np.array([p["N"] for p in pts], float)
        y = np.log([p["off"]["Etau"] for p in pts])
        w = np.array([p["off"]["n_exit"] for p in pts], float)
        refits = []
        for model in ("lin", "logN", "invN"):
            for drop in (0, 1, 2):
                m = np.arange(len(Np)) >= drop
                cols = model_cols(Np[m], model)
                if m.sum() - len(cols) < 1:
                    continue
                sol, cov, dof, chi2 = wls(cols, y[m], w[m])
                red = chi2 / dof
                se_nom = float(np.sqrt(cov[0, 0]))
                se = se_nom * max(1.0, red) ** 0.5
                row = dict(model=model, dropped=drop, n=int(m.sum()), dof=dof, dV=float(sol[0]),
                           se_nominal=se_nom, se=se, chi2_red=float(red),
                           chi2_p=float(stats.chi2.sf(chi2, dof)))
                for name, ref in (("num", dV), ("inf", DV_INF)):
                    t = (sol[0] - ref) / se
                    row[f"t_vs_{name}"] = float(t)
                    row[f"p_normal_vs_{name}"] = float(2 * stats.norm.sf(abs((sol[0] - ref) / se_nom)))
                    row[f"p_student_vs_{name}"] = float(2 * stats.t.sf(abs(t), dof))
                refits.append(row)
        R["ssa_refits"] = refits
        # the same three models fitted to the backward equation AT THE SIMULATED SIZES:
        # what an infinitely long simulation campaign would have returned
        exact = []
        yb = np.array([np.log(tab["off"][int(N)]["tau"]) for N in Np])
        for model in ("lin", "logN", "invN"):
            if len(Np) - len(model_cols(Np, model)) < 1:
                continue
            sol, *_ = wls(model_cols(Np, model), yb, w)
            exact.append(dict(model=model, dV=float(sol[0]), dV_minus_num=float(sol[0] - dV)))
        R["bke_at_simulated_sizes"] = exact
        out[key] = R

    summary = {}
    for k, v in zs.items():
        v = np.array(v, float)
        if len(v):
            summary[k] = dict(n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)),
                              max_abs=float(np.abs(v).max()), chi2=float(np.sum(v**2)),
                              chi2_p=float(stats.chi2.sf(np.sum(v**2), len(v))),
                              mean_p=float(2 * stats.norm.sf(abs(v.mean()) * np.sqrt(len(v)))))
    (BKE / "analysis.json").write_text(json.dumps(dict(rates=out, ssa_vs_bke=summary), indent=1))

    # ---- printed summary
    print("SIMULATIONS AGAINST THE BACKWARD EQUATION (standardized differences of log means)")
    for k, s in summary.items():
        print(f"  {k:14s} n={s['n']:3d} mean={s['mean']:+.3f} (p={s['mean_p']:.2f}) sd={s['sd']:.3f} "
              f"max|z|={s['max_abs']:.2f} chi2={s['chi2']:.1f} (p={s['chi2_p']:.2f})")
    for key, R in out.items():
        dV = R["dV_num"]
        print(f"\nr = {key}   dV_num = {dV:.6f}")
        pf = dict(zip(R["N"], R["prefactor"]))
        print("  prefactor:", "  ".join(f"N={N}: {pf[N]:.4f}" for N in (40, 60, 100, 140, 200, 300, 400) if N in pf))
        ls = R["local_slopes"]
        print("  local slopes:", "  ".join(f"{s['N_lo']}-{s['N_hi']}: {s['slope']:.6f}" for s in (ls[0], ls[len(ls) // 2], ls[-3], ls[-1])))
        print("  direct/renewal-1 (largest):", max(R["direct_over_renewal_minus_1"].items(), key=lambda kv: abs(kv[1]), default=None))
        for f in R["bke_fits"]:
            if f["model"] != "lin" and f["N_min"] in (40, 120, 200):
                print(f"  BKE fit {f['model']:6s} N>={f['N_min']:3d}: dV={f['dV']:.6f} ({f['dV_minus_num']:+.1e})  "
                      f"prefactor->{f['prefactor_limit']:.4f}  c={f['c']:.2f}  rms={f['rms_resid']:.1e}")
        for f in R["bke_at_simulated_sizes"]:
            print(f"  exact means at the simulated sizes, model {f['model']:5s}: dV={f['dV']:.5f} ({f['dV_minus_num']:+.5f})")
        for f in R["ssa_refits"]:
            print(f"  SSA {f['model']:5s} drop {f['dropped']}: dV={f['dV']:.4f}+-{f['se']:.4f} chi2red={f['chi2_red']:.2f}({f['dof']}) "
                  f"vs num t={f['t_vs_num']:+.1f} p_t={f['p_student_vs_num']:.2g} | vs inf t={f['t_vs_inf']:+.1f} p_t={f['p_student_vs_inf']:.2g}")
        for f in R["thr_fits_with_lattice_term"]:
            print(f"  thr (BKE) N>={f['N_min']:3d}: dV_thr={f['dV_thr']:.5f} (ref {R['thr_reference']:.5f}) g={f['g']:.3f} c={f['c']:.1f} rms={f['rms_resid']:.1e}")


if __name__ == "__main__":
    main()
