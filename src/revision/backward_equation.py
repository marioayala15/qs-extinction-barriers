"""Referee point 3: mean exit times of the four-channel network from the backward equation.

For the jump process at population scale N (c = 0.36), started at the lattice point u0
nearest to N * U_on, this module computes E[tau] for three stopping sets, without sampling
and without the action solver:

    off    first entry into R_off = [0, 0.30] x [0, 0.70]        (the extinction time)
    small  first entry into       [0, 0.15] x [0, 0.35]
    thr    first time X <= x*                                    (the threshold time)

The state space is truncated to [0, xmax] x [0, wmax] with reflecting truncation (division
is switched off at xmax, production at wmax); the box is U_on plus BOX_SD stationary
standard deviations of the Gaussian regime, and is varied as a check.

TWO SOLVERS.

  direct    solve  -Q tau = 1  outside the target.  The matrix has an eigenvalue of size
            1/E[tau], so the rounding error grows with E[tau]; it is used only as a
            cross-check while E[tau] is moderate.

  renewal   remove u0 from the domain as well.  With
                h(u) = P_u(target before u0),      -Q h = (rate into the target),
                m(u) = E_u[time to target or u0],  -Q m = 1,
            both on the domain without u0 and the target, the strong Markov property at
            the returns to u0 gives the exact identity
                E_u0[tau] = ( 1/q0 + sum_j P(u0 -> j) m(j) ) / ( sum_j P(u0 -> j) h(j) ),
            q0 the total rate at u0.  Removing u0 removes the small eigenvalue (the process
            returns to u0 quickly), so both systems are well conditioned and h, which is
            of order exp(-N dV), is obtained to high relative accuracy.

    python src/revision/backward_equation.py --check     # solver checks, about two minutes
    python src/revision/backward_equation.py --write     # the full grid
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_continuous_lyapunov
from scipy.sparse.linalg import splu

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "collapse_time"))
from ldp_action import PAR as P, hill, dhill  # noqa: E402
from common import C_COST, RECT_LARGE, RECT_SMALL, fixed_points, initial_counts  # noqa: E402

OUT = ROOT / "data" / "revision" / "backward_equation"
RATES = (0.5, 1.0, 2.0, 4.0, 8.0)
BOX_SD = 9.0
X_STAR, X_ON = fixed_points(C_COST)
W_ON = P["aC"] * X_ON / P["kappa"]


def stationary_sd(r, c=C_COST):
    """Standard deviations of sqrt(N)(U^N - U_on) in the Gaussian regime."""
    b = P["b0"] + P["b1"] * hill(W_ON, P["Kh"], P["h"])
    db = P["b1"] * dhill(W_ON, P["Kh"], P["h"])
    J = np.array([[-P["theta"] * X_ON, X_ON * db],
                  [r * P["aC"], -r * P["kappa"]]])
    D = np.diag([2 * X_ON * b, 2 * r * P["aC"] * X_ON])
    S = solve_continuous_lyapunov(J, -D)
    return float(np.sqrt(S[0, 0])), float(np.sqrt(S[1, 1]))


def box(N, r, sd=BOX_SD):
    sx, sw = stationary_sd(r)
    return X_ON + sd * sx / np.sqrt(N), W_ON + sd * sw / np.sqrt(N)


def lattice(N, r, xmax, wmax, c=C_COST):
    nc, nw = int(np.ceil(xmax * N)), int(np.ceil(wmax * N))
    I, J = np.meshgrid(np.arange(nc + 1), np.arange(nw + 1), indexing="ij")
    w, x = J / N, I / N
    div = I * (P["b0"] + P["b1"] * hill(w, P["Kh"], P["h"]))
    dth = I * (P["d0"] + c + P["theta"] * x)
    pro = r * P["aC"] * I * 1.0
    rem = r * P["kappa"] * J * 1.0
    div[nc, :] = 0.0
    pro[:, nw] = 0.0
    return I, J, ((div, 1, 0), (dth, -1, 0), (pro, 0, 1), (rem, 0, -1))


def target_mask(I, J, N, which):
    if which == "off":
        a, b_ = RECT_LARGE
    elif which == "small":
        a, b_ = RECT_SMALL
    elif which == "thr":
        return I <= X_STAR * N                       # the simulator's comparison
    else:
        raise ValueError(which)
    return (I <= np.floor(a * N + 1e-9)) & (J <= np.floor(b_ * N + 1e-9))


def assemble(I, J, chan, dom):
    """-Q restricted to `dom` (CSC), the index map, and the rate from each state of `dom`
    into the target (the complement of dom, u0 excepted, is handled by the caller)."""
    n = int(dom.sum())
    idx = -np.ones(I.shape, dtype=np.int64)
    idx[dom] = np.arange(n)
    tot = sum(c[0] for c in chan)
    rows, cols, vals = [np.arange(n)], [np.arange(n)], [tot[dom]]
    for rate, di, dj in chan:
        m = dom & (rate > 0)
        to = idx[I[m] + di, J[m] + dj]
        ok = to >= 0
        rows.append(idx[m][ok]); cols.append(to[ok]); vals.append(-rate[m][ok])
    A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n))
    return A, idx


def refine(lu, A, b, x, steps=1):
    for _ in range(steps):
        x = x + lu.solve(b - A @ x)
    return x


def solve(N, r, which="off", sd=BOX_SD, direct=False, permc="COLAMD"):
    """E[tau] from u0 by the renewal identity; with direct=True also by the direct solve."""
    t0 = time.time()
    xmax, wmax = box(N, r, sd)
    I, J, chan = lattice(N, r, xmax, wmax)
    tgt = target_mask(I, J, N, which)
    i0, j0 = (int(v) for v in initial_counts(N))
    out = dict(N=N, r=r, target=which, box_sd=sd, xmax=xmax, wmax=wmax)

    dom = ~tgt
    dom[i0, j0] = False
    A, idx = assemble(I, J, chan, dom)
    into_tgt = np.zeros(I.shape)
    for rate, di, dj in chan:                      # rate from each state into the target
        m = dom & (rate > 0)
        hit = np.zeros(I.shape, bool)
        hit[m] = tgt[I[m] + di, J[m] + dj]
        into_tgt[hit] += rate[hit]
    lu = splu(A, permc_spec=permc)
    bh = into_tgt[dom]
    h = refine(lu, A, bh, lu.solve(bh))
    one = np.ones(A.shape[0])
    m_ = refine(lu, A, one, lu.solve(one))
    q0 = sum(float(c[0][i0, j0]) for c in chan)
    p_esc, cyc = 0.0, 1.0 / q0
    for rate, di, dj in chan:
        k = idx[i0 + di, j0 + dj]
        p_esc += rate[i0, j0] / q0 * h[k]
        cyc += rate[i0, j0] / q0 * m_[k]
    out.update(tau=cyc / p_esc, p_escape=p_esc, cycle_time=cyc, unknowns=int(A.shape[0]),
               resid_h=float(np.max(np.abs(A @ h - bh)) / max(np.max(np.abs(bh)), 1e-300)),
               resid_m=float(np.max(np.abs(A @ m_ - one))))
    del lu

    if direct:
        dom = ~tgt
        A, idx = assemble(I, J, chan, dom)
        lu = splu(A, permc_spec=permc)
        one = np.ones(A.shape[0])
        out["tau_direct"] = float(lu.solve(one)[idx[i0, j0]])
        del lu
    out["secs"] = time.time() - t0
    return out


def _task(args):
    N, r, which, direct = args
    res = solve(N, r, which, direct=direct)
    print(f"r={r:g} N={N} {which}: tau={res['tau']:.8g}"
          + (f" direct={res['tau_direct']:.8g}" if direct else "")
          + f" unknowns={res['unknowns']} {res['secs']:.0f}s", flush=True)
    return res


def n_grid(r, nmax):
    ssa = json.loads((ROOT / "data" / "collapse_time" / "fit_results.json").read_text())
    stored = ssa["rates"][f"{r:g}"]["Ns"]
    return sorted(set(stored) | set(range(40, 161, 10)) | set(range(180, nmax + 1, 20)))


def check():
    print("renewal against direct, and box 9 sd against 12 sd")
    for r in RATES:
        for N in (40, 80, 120):
            a = solve(N, r, "off", direct=True)
            b = solve(N, r, "off", sd=12.0)
            c = solve(N, r, "off", permc="MMD_AT_PLUS_A")
            print(f"r={r:g} N={N}: tau={a['tau']:.10g}  direct/renewal-1={a['tau_direct'] / a['tau'] - 1:+.1e}"
                  f"  box12/box9-1={b['tau'] / a['tau'] - 1:+.1e}  ordering={c['tau'] / a['tau'] - 1:+.1e}"
                  f"  resid_h={a['resid_h']:.0e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--nmax", type=int, default=400)
    ap.add_argument("--workers-small", type=int, default=8)
    ap.add_argument("--workers-large", type=int, default=3)
    args = ap.parse_args()
    if args.check:
        check()
    if not args.write:
        return
    dv = json.loads((ROOT / "data" / "collapse_time" / "fit_results.json").read_text())["rates"]
    tasks = []
    for r in RATES:
        dV = dv[f"{r:g}"]["saddle_dV_num"]
        for N in n_grid(r, args.nmax):
            for which in ("off", "thr", "small"):
                tasks.append((N, r, which, which == "off" and N * dV <= 21.0))
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for lo, hi, workers in ((0, 160, args.workers_small), (161, 10**9, args.workers_large)):
        part = sorted((t for t in tasks if lo <= t[0] <= hi), key=lambda t: t[0])
        with Pool(workers) as pool:
            for res in pool.imap_unordered(_task, part, chunksize=1):
                rows.append(res)
                (OUT / "results.json").write_text(json.dumps(
                    dict(settings=dict(c=C_COST, box_sd=BOX_SD, x_on=X_ON, w_on=W_ON,
                                       x_star=X_STAR), rows=rows), indent=1))
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
