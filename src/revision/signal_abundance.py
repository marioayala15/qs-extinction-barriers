"""Referee point 4: signal lag against signal noise.

The signal count is scaled as K N instead of N: signal jumps (0, +-1/K) at rates
K r aC x and K r kappa w.  The drift is unchanged, the signal part of the Lagrangian is
multiplied by K, and K -> infinity is the hybrid model with a deterministic signal.

First-order prediction (same boundary-layer algebra as for A):

    A(K) = A_lag - (A_lag - A) / K,
    A_lag = (aC/kappa^2) int M dx,   A = A_lag - (aC/kappa^2) int x M^2 / F dx,
    M = b'(wbar) F / b(wbar).

The script compares r [dV_K(r) - dV_inf] at large r with A(K).

    python src/revision/signal_abundance.py        # about two minutes
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.integrate import quad

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ldp_action import PAR as P, Network, mam_action, net_wellmixed_explicit  # noqa: E402
from ldp_crossover_sweep import pieces, Vinf, C_WORK, T_HORIZON, M_NODES, MAXITER  # noqa: E402

KS = (1.0, 2.0, 4.0, 16.0, 64.0)
RS = (8.0, 16.0, 32.0, 64.0)


def net_K(c, r, K):
    base = net_wellmixed_explicit(c, rN=r)
    jumps = base.D.copy()
    jumps[2:] /= K
    scale = np.array([1.0, 1.0, K, K])
    return Network(jumps, lambda U: base.rates(U) * scale,
                   lambda U: base.jac(U) * scale[:, None], name=f"K={K} r={r}")


def coefficients(c):
    b, db, wb, d, F, x_st, x_on = pieces(c)
    M = lambda x: db(wb(x)) * F(x) / b(wb(x))
    pre = P["aC"] / P["kappa"] ** 2
    A_lag = pre * quad(M, x_st, x_on)[0]
    A = A_lag - pre * quad(lambda x: x * M(x) ** 2 / F(x), x_st, x_on)[0]
    return A, A_lag


def run(args):
    K, r = args
    c = C_WORK
    b, db, wb, d, F, x_st, x_on = pieces(c)
    t0 = time.time()
    res = mam_action(net_K(c, r, K), [x_on, wb(x_on)], [x_st, wb(x_st)], T=T_HORIZON,
                     K=M_NODES, maxiter=MAXITER)
    return dict(K=K, r=r, dV=res["S"], success=res["success"], secs=time.time() - t0)


if __name__ == "__main__":
    c = C_WORK
    A, A_lag = coefficients(c)
    V = Vinf(c)
    print(f"A = {A:.6f}   A_lag = {A_lag:.6f}   dV_inf = {V:.7f}", flush=True)
    with Pool(8) as pool:
        rows = pool.map(run, [(K, r) for K in KS for r in RS], chunksize=1)
    for row in rows:
        row["plateau"] = row["r"] * (row["dV"] - V)
        row["A_K"] = A_lag - (A_lag - A) / row["K"]
    for K in KS:
        sub = [x for x in rows if x["K"] == K]
        print(f"K={K:g}: A(K)={sub[0]['A_K']:.6f}  plateau " +
              "  ".join(f"r={x['r']:g}: {x['plateau']:.6f}" for x in sub) +
              f"   rel diff at r=64: {100 * (sub[-1]['plateau'] / sub[-1]['A_K'] - 1):+.2f}%"
              f"  ok={all(x['success'] for x in sub)}", flush=True)
    out = ROOT / "data" / "revision" / "signal_abundance.json"
    out.write_text(json.dumps(dict(c=c, A=A, A_lag=A_lag, dV_inf=V, rows=rows), indent=1))
    print("wrote", out)
