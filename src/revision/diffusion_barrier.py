"""Referee point 6: the barrier of the diffusion approximation at finite r.

The jump Hamiltonian H(U,p) = sum_rho beta_rho (exp<p,Delta_rho> - 1) is replaced by its
second-order truncation in p,

    H_d(U,p) = <F(U),p> + (1/2) <p, D(U) p>,

i.e. the Hamiltonian of the diffusion with drift F and diffusion matrix D of the Gaussian
regime.  The same discretisation and optimiser (`ldp_action.mam_action`, T = 40, M = 2000,
maxiter = 60000) then give the diffusion barrier from U_on to U_*, to be compared with the
jump barrier in data/crossover_barriers.json.

Check: for the eliminated-signal problem the diffusion barrier has the closed form
int 2 (b - d)/(b + d) dx.

    python src/revision/diffusion_barrier.py            # about one minute
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ldp_action import Network, mam_action, net_1d_slaved, net_wellmixed_explicit  # noqa: E402
from ldp_crossover_sweep import pieces, C_WORK, R_GRID, T_HORIZON, M_NODES, MAXITER  # noqa: E402

OUT = ROOT / "data" / "revision" / "diffusion_barrier.json"


class DiffusionNetwork(Network):
    """Same drift and diffusion matrix as `net`, quadratic Hamiltonian in p."""

    def __init__(self, net):
        super().__init__(net.D, net.rates, net.jac, name="diffusion " + net.name)

    def _s(self, p):
        return np.einsum("...n,rn->...r", p, self.D)

    def H(self, U, p):
        s = self._s(p)
        return np.einsum("...r,...r->...", self.rates(U), s + 0.5 * s * s)

    def gradp_H(self, U, p):
        return np.einsum("...r,...r,rn->...n", self.rates(U), 1.0 + self._s(p), self.D)

    def hessp_H(self, U, p):
        return np.einsum("...r,rn,rm->...nm", self.rates(U), self.D, self.D)

    def gradU_H(self, U, p):
        s = self._s(p)
        return np.einsum("...rn,...r->...n", self.jac(U), s + 0.5 * s * s)


def main():
    c = C_WORK
    b, db, wb, d, F, x_st, x_on = pieces(c)
    U_on, U_st = [x_on, wb(x_on)], [x_st, wb(x_st)]
    jump = {row["r"]: row["dV"] for row in
            json.loads((ROOT / "data" / "crossover_barriers.json").read_text())["costs"][f"{c:g}"]["rows"]}

    closed = quad(lambda x: 2 * (b(wb(x)) - d(x)) / (b(wb(x)) + d(x)), x_st, x_on)[0]
    t0 = time.time()
    one = mam_action(DiffusionNetwork(net_1d_slaved(c)), [x_on], [x_st], T=T_HORIZON, K=M_NODES,
                     maxiter=MAXITER)
    print(f"eliminated signal: diffusion MAM {one['S']:.7f}, closed form {closed:.7f}, "
          f"success {one['success']}, {time.time() - t0:.1f}s", flush=True)

    rows = []
    for r in R_GRID:
        t0 = time.time()
        res = mam_action(DiffusionNetwork(net_wellmixed_explicit(c, rN=r)), U_on, U_st,
                         T=T_HORIZON, K=M_NODES, maxiter=MAXITER)
        # momentum along the jump minimiser is not stored; report the diffusion value only
        rows.append(dict(r=r, dV_diffusion=res["S"], dV_jump=jump[r],
                         rel_diff=res["S"] / jump[r] - 1.0, success=res["success"],
                         secs=time.time() - t0))
        print(f"r={r:g}: diffusion {res['S']:.7f}  jump {jump[r]:.7f}  "
              f"rel diff {100 * rows[-1]['rel_diff']:+.4f}%  success {res['success']}  "
              f"{rows[-1]['secs']:.1f}s", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        settings=dict(c=c, T=T_HORIZON, M=M_NODES, maxiter=MAXITER),
        eliminated=dict(mam=one["S"], closed_form=closed, success=one["success"]),
        rows=rows), indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
