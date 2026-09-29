"""Robustness of the minimum-action barriers (action plan N1-N4).

N1  multi-start: for r in R_SET the saddle problem U_on -> U_* is solved from four initial
    paths: the straight segment on w = wbar(x) (production), the same segment bowed up and
    down in w by +-BOW (w_on - w_*) sin(pi s), and the lifted eliminated-signal instanton
    (x following dx/dt = -F(x) on w = wbar(x), resampled to the mesh).
N2  Hamiltonian residual: H(Ubar_k, p_k) with p_k the Legendre momentum of each interval.
    A free-time minimiser has H = 0; at fixed T the discrete minimiser has H roughly
    constant and small.
N3  exact action of the piecewise-linear interpolant, each interval integrated by
    Gauss-Legendre (GL nodes) at its constant velocity.  Any admissible path's action
    bounds the infimum from above; the midpoint sum does not.
N4  settings as in ldp_crossover_sweep (T = 40, M = 2000, maxiter = 60000); L-BFGS-B
    ftol = 1e-14, gtol = 1e-10 (ldp_action.mam_action).  A start that stops on the iteration
    cap is warm-restarted from its last path, at most MAX_RESTARTS times, then raises.

Writes data/robustness/mam_robustness.json and the final paths to mam_robustness_paths.npz.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from ldp_action import lagrangian, mam_action, net_wellmixed_explicit  # noqa: E402
from ldp_crossover_sweep import C_WORK, M_NODES, MAXITER, T_HORIZON, pieces  # noqa: E402

OUT = SRC.parent / "data" / "robustness"
R_SET = (0.5, 1.0, 4.0, 16.0, 64.0)
BOW = 0.5
GL = 8
MAX_RESTARTS = 3


def initial_paths(c, K):
    _, _, wb, _, F, x_st, x_on = pieces(c)
    s = np.linspace(0.0, 1.0, K + 1)
    UA, UB = np.array([x_on, wb(x_on)]), np.array([x_st, wb(x_st)])
    straight = UA[None] * (1 - s[:, None]) + UB[None] * s[:, None]
    bump = np.zeros_like(straight)
    bump[:, 1] = BOW * (UA[1] - UB[1]) * np.sin(np.pi * s)
    # 1-D instanton: dx/dt = -F(x) leaves x_on and approaches x_st; start/stop 1e-4 inside
    eps = 1e-4 * (x_on - x_st)
    sol = solve_ivp(lambda t, x: -F(x), (0, 1e4), [x_on - eps], max_step=0.05, dense_output=True,
                    events=lambda t, x: x[0] - (x_st + eps), rtol=1e-10, atol=1e-12)
    t_end = sol.t_events[0][0]
    x = sol.sol(s * t_end)[0]
    x[0], x[-1] = x_on, x_st
    inst = np.stack([x, wb(x)], axis=1)
    return UA, UB, dict(straight=straight, bowed_up=straight + bump, bowed_down=straight - bump,
                        instanton_1d=inst)


def hausdorff(P, Q):
    from scipy.spatial.distance import directed_hausdorff
    return float(max(directed_hausdorff(P, Q)[0], directed_hausdorff(Q, P)[0]))


def diagnostics(net, P, T):
    K = P.shape[0] - 1
    h = T / K
    V = (P[1:] - P[:-1]) / h
    Ubar = 0.5 * (P[:-1] + P[1:])
    Lmid, p = lagrangian(net, Ubar, V)
    Hk = net.H(Ubar, p)
    xg, wg = np.polynomial.legendre.leggauss(GL)
    Lsum = 0.0
    for g, wt in zip(xg, wg):
        Ug = P[:-1] + 0.5 * (1 + g) * (P[1:] - P[:-1])
        Lg, _ = lagrangian(net, Ug, V, p0=p)
        Lsum += 0.5 * wt * float(np.sum(Lg))
    return dict(S_midpoint=float(np.sum(Lmid) * h), S_exact_pl=Lsum * h,
                H_max_abs=float(np.max(np.abs(Hk))), H_median=float(np.median(Hk)),
                H_range=[float(Hk.min()), float(Hk.max())],
                H_max_abs_inner90=float(np.max(np.abs(Hk[K // 20: K - K // 20]))))


def main():
    stored = json.loads((SRC.parent / "data" / "crossover_barriers.json").read_text())
    dV = {row["r"]: row["dV"] for row in stored["costs"][f"{C_WORK:g}"]["rows"]}
    out = dict(settings=dict(c=C_WORK, T=T_HORIZON, M=M_NODES, maxiter=MAXITER, ftol=1e-14, gtol=1e-10,
                             bow=BOW, gauss_legendre_nodes=GL), rates={})
    paths = {}
    for r in R_SET:
        net = net_wellmixed_explicit(C_WORK, rN=r)
        UA, UB, starts = initial_paths(C_WORK, M_NODES)
        R = dict(stored_dV=dV.get(r), starts={})
        final = {}
        for name, P0 in starts.items():
            t0 = time.time()
            res = mam_action(net, UA, UB, T=T_HORIZON, K=M_NODES, path0=P0, maxiter=MAXITER)
            restarts = 0
            while not res["success"] and restarts < MAX_RESTARTS:   # iteration cap: warm restart
                restarts += 1
                res = mam_action(net, UA, UB, T=T_HORIZON, K=M_NODES, path0=res["path"], maxiter=MAXITER)
            if not res["success"]:
                raise RuntimeError(f"not converged: r={r}, start={name}, {restarts} restarts")
            final[name] = res["path"]
            d = diagnostics(net, res["path"], T_HORIZON)
            d.update(S=res["S"], secs=time.time() - t0, warm_restarts=restarts,
                     S_initial=diagnostics(net, P0, T_HORIZON)["S_midpoint"],
                     # at fixed T, dwell near the endpoints is almost free, so compare the
                     # traces as point sets (Hausdorff), not at equal times
                     hausdorff_to_straight_result=hausdorff(res["path"], final["straight"]))
            R["starts"][name] = d
            print(f"r={r:g} {name:12s} S={res['S']:.7f} exactPL={d['S_exact_pl']:.7f} "
                  f"max|H|={d['H_max_abs']:.1e} hausdorff={d['hausdorff_to_straight_result']:.1e} "
                  f"restarts={restarts} ({d['secs']:.0f}s)", flush=True)
        S = [v["S"] for v in R["starts"].values()]
        R["spread"] = max(S) - min(S)
        R["gap_exact_minus_midpoint"] = {k: v["S_exact_pl"] - v["S"] for k, v in R["starts"].items()}
        out["rates"][f"{r:g}"] = R
        paths[f"{r:g}"] = final
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "mam_robustness_paths.npz",
                        **{f"r{r}_{name}": P for r, fin in paths.items() for name, P in fin.items()})
    (OUT / "mam_robustness.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
