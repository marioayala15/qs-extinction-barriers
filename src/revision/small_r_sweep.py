"""Referee point 5: minimum-action barriers below r = 0.5.

The signal relaxes on the time 1/(r kappa), so the horizon is scaled as T = T_SCALE / r
(T = 40 at r = 0.5 is T_SCALE = 20).  For every rate the action is minimised on a sequence
of meshes dt = 0.08, 0.04, 0.02, each warm-started from the previous one, at two horizons,
so that horizon and mesh effects can be read separately, as in ldp_crossover_sweep --check.

Reference for the limit r -> 0 (formal elimination of the cells, x = (b(w) - d0 - c)/theta):
    V_0 = int_{w*}^{w_on} log( aC x(w) / (kappa w) ) dw.

    python src/revision/small_r_sweep.py            # all rates, one process each
    python src/revision/small_r_sweep.py 0.25       # one rate
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
from ldp_action import PAR as P, hill, mam_action, net_wellmixed_explicit  # noqa: E402
from ldp_crossover_sweep import pieces, C_WORK  # noqa: E402

OUT = ROOT / "data" / "revision" / "small_r"
R_GRID = (0.5, 0.5 / 2**0.5, 0.25, 0.25 / 2**0.5, 0.125, 0.125 / 2**0.5, 0.0625, 0.03125)
T_SCALES = (20.0, 30.0)
DTS = (0.08, 0.04, 0.02)
MAXITER = 400000


def slow_signal_limit(c=C_WORK):
    b, db, wb, d, F, x_st, x_on = pieces(c)
    xt = lambda w: (b(w) - P["d0"] - c) / P["theta"]
    return quad(lambda w: np.log(P["aC"] * xt(w) / (P["kappa"] * w)), wb(x_st), wb(x_on))[0]


def run_rate(r):
    c = C_WORK
    b, db, wb, d, F, x_st, x_on = pieces(c)
    U_on, U_st = np.array([x_on, wb(x_on)]), np.array([x_st, wb(x_st)])
    net = net_wellmixed_explicit(c, rN=r)
    rows = []
    for ts in T_SCALES:
        T = ts / r
        path = None
        for dt in DTS:
            K = int(round(T / dt))
            if path is not None:                       # refine the previous minimiser
                s_old = np.linspace(0.0, 1.0, path.shape[0])
                s_new = np.linspace(0.0, 1.0, K + 1)
                path = np.column_stack([np.interp(s_new, s_old, path[:, i]) for i in range(2)])
            t0 = time.time()
            res = mam_action(net, U_on, U_st, T=T, K=K, path0=path, maxiter=MAXITER)
            path = res["path"]
            rows.append(dict(r=r, T=T, T_scale=ts, dt=dt, K=K, dV=res["S"],
                             success=res["success"], secs=time.time() - t0,
                             w_max=float(path[:, 1].max()), x_min=float(path[:, 0].min())))
            print(f"r={r:.5f} T={T:.0f} dt={dt}: dV={res['S']:.7f} success={res['success']} "
                  f"{rows[-1]['secs']:.0f}s", flush=True)
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / f"r{r:.5f}.json").write_text(json.dumps(rows, indent=1))
            if dt == DTS[-1] and ts == T_SCALES[0]:
                np.save(OUT / f"path_r{r:.5f}.npy", path[::max(1, K // 2000)])
    return rows


if __name__ == "__main__":
    print(f"slow-signal reduced barrier V_0 = {slow_signal_limit():.6f}", flush=True)
    rates = [float(a) for a in sys.argv[1:]] or list(R_GRID)
    with Pool(min(len(rates), 8)) as pool:
        allrows = pool.map(run_rate, rates, chunksize=1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(
        dict(c=C_WORK, V0=slow_signal_limit(), rows=[x for rr in allrows for x in rr]), indent=1))
    print("wrote", OUT / "summary.json")
