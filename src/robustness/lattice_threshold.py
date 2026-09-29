"""Direct lattice test of the threshold rounding effect (action plan S3).

The threshold time stops at the first X <= x* N, i.e. at the lattice count floor(N x*).
Here the stop is moved to X <= floor(N x*) - k, k = -1, ..., 3, at fixed (r, N), with the
compiled SSA in mode "thr" (src/collapse_time/ssa_offtarget).  If the oscillating term in
the threshold fits comes from rounding, log E[tau] should rise by about the fitted
coefficient of frac(N x*) per lattice step: 0.27 at r = 0.5 and 0.23 at r = 1 (campaign 1).

Usage:  python lattice_threshold.py [--workers 10] [--nrep 4000]
Writes data/robustness/lattice_threshold.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "collapse_time"))
from common import (BIN, C_COST, ON_BOX, PAPER, RECT_LARGE, RECT_SMALL, censored_mle,  # noqa: E402
                    fixed_points, initial_counts, saddle_actions)

OUT = PAPER / "data" / "robustness"
PLAN = [(r, N) for r in (0.5, 1.0) for N in (60, 80)]
KS = (-1, 0, 1, 2, 3)
N_CHUNK = 20
K_H = 50.0
SEED_TAG = 2026092900          # distinct from 20260812 and 2026091600


def run_chunk(r, N, k, chunk, nrep, H):
    x_star, x_on = fixed_points()
    X0, W0 = initial_counts(N)
    xthr = (np.floor(N * x_star) - k + 0.5) / N
    with tempfile.TemporaryDirectory() as td:
        ft, fa = f"{td}/t.bin", f"{td}/a.bin"
        cmd = [str(BIN), "thr", str(N), repr(r), repr(C_COST), str(X0), str(W0),
               repr(float(xthr)), repr(H), str(nrep), str(SEED_TAG + 100 * k + chunk),
               str(int(round(100 * r))), str(N),
               *map(repr, (*RECT_LARGE, *RECT_SMALL, x_on, 2 * x_on, *ON_BOX)), ft, fa]
        t0 = time.time()
        subprocess.run(cmd, check=True)
        secs = time.time() - t0
        traj = np.fromfile(ft).reshape(-1, 12)
    return (r, N, k), traj, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--nrep", type=int, default=4000)
    args = ap.parse_args()
    x_star, _ = fixed_points()
    dV = saddle_actions()
    per = args.nrep // N_CHUNK
    tasks = [(r, N, k, ch, per, K_H * float(np.exp(N * dV[r])))
             for r, N in PLAN for k in KS for ch in range(N_CHUNK)]
    tasks.sort(key=lambda t: -t[1] * np.exp(t[1] * dV[t[0]]))
    t_start = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(lambda t: run_chunk(*t), tasks))
    print(f"wall {time.time() - t_start:.0f}s", flush=True)

    out = dict(settings=dict(c=C_COST, x_star=x_star, ks=KS, nrep=per * N_CHUNK, K_H=K_H,
                             seed_tag=SEED_TAG, stop="first X <= floor(N x*) - k",
                             prediction_per_step={"0.5": 0.269, "1": 0.226}),
               points=[])
    for r, N in PLAN:
        H = K_H * float(np.exp(N * dV[r]))
        row = dict(r=r, N=N, floor_Nxstar=int(np.floor(N * x_star)),
                   frac_Nxstar=float(N * x_star - np.floor(N * x_star)), horizon=H, k={})
        for k in KS:
            traj = np.vstack([t for key, t, _ in res if key == (r, N, k)])
            X = traj[:, 1][np.isfinite(traj[:, 0])]
            E, n = censored_mle(traj[:, 0], np.isfinite(traj[:, 0]), H)
            row["k"][k] = dict(stop_count=int(np.floor(N * x_star)) - k, Etau=E, n_exit=n,
                               n_rep=len(traj), se_log=float(1 / np.sqrt(n)),
                               X_at_stop_ok=bool(np.all(X == np.floor(N * x_star) - k)),
                               core_secs=float(sum(s for key, _, s in res if key == (r, N, k))))
        for k in KS[1:]:
            a, b = row["k"][k], row["k"][k - 1]
            row["k"][k]["dlogE_per_step"] = float(np.log(a["Etau"] / b["Etau"]))
            row["k"][k]["se"] = float(np.hypot(a["se_log"], b["se_log"]))
        out["points"].append(row)
        print(f"r={r:g} N={N} floor(Nx*)={row['floor_Nxstar']}: " + ", ".join(
            f"k={k} E={row['k'][k]['Etau']:.1f}" + (f" d={row['k'][k]['dlogE_per_step']:.3f}±{row['k'][k]['se']:.3f}"
                                                  if k != KS[0] else "") for k in KS), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lattice_threshold.json").write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
