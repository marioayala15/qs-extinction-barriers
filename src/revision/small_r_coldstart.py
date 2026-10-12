"""Cold-start check for small_r_sweep.py: at r = 0.0625 and 0.125 (T = 20/r, dt = 0.04) the
minimisation is repeated from the straight segment bowed up and down in w, with no mesh
sequence.  The values should agree with the warm-started ones in data/revision/small_r/."""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from ldp_action import mam_action, net_wellmixed_explicit  # noqa: E402
from ldp_crossover_sweep import pieces, C_WORK  # noqa: E402
from small_r_sweep import MAXITER  # noqa: E402


def run(args):
    r, bow = args
    b, db, wb, d, F, x_st, x_on = pieces(C_WORK)
    U_on, U_st = np.array([x_on, wb(x_on)]), np.array([x_st, wb(x_st)])
    T, dt = 20.0 / r, 0.04
    K = int(round(T / dt))
    s = np.linspace(0.0, 1.0, K + 1)[:, None]
    path = U_on * (1 - s) + U_st * s
    path[:, 1] += bow * np.sin(np.pi * s[:, 0])
    t0 = time.time()
    res = mam_action(net_wellmixed_explicit(C_WORK, rN=r), U_on, U_st, T=T, K=K, path0=path,
                     maxiter=MAXITER)
    out = dict(r=r, bow=bow, T=T, dt=dt, dV=res["S"], success=res["success"],
               secs=time.time() - t0)
    print(out, flush=True)
    return out


if __name__ == "__main__":
    jobs = [(r, bow) for r in (0.125, 0.0625) for bow in (0.6, -0.6)]
    with Pool(4) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    out = ROOT / "data" / "revision" / "small_r" / "coldstart.json"
    out.write_text(json.dumps(rows, indent=1))
    print("wrote", out)
