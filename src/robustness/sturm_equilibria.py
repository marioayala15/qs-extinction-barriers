"""Exact count of the equilibria of the mean-field system (action plan M7).

The drift is F(x, w) = ( x (b(w) - d(x)),  r (aC x - kappa w) ), with
b(w) = b0 + b1 w^h / (Kh^h + w^h) and d(x) = d0 + c + theta x.  An equilibrium has
w = aC x / kappa, and either x = 0 or b(aC x / kappa) = d(x).  For x > 0, multiplying the
second equation by Kh^h + (aC x / kappa)^h (which is positive) gives the polynomial

    P(x) = (b0 - d0 - c - theta x) (Kh^h + (aC x / kappa)^h) + b1 (aC x / kappa)^h,

of degree h + 1 = 5.  So the positive equilibria are exactly the positive roots of P.  They are
counted with a Sturm sequence in exact rational arithmetic (parameters read from PAR as
decimals and converted exactly), and gcd(P, P') = 1 shows every root is simple.
The isolating intervals are refined by exact bisection and compared with the float root search.

Writes data/robustness/sturm_equilibria.json.
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction as Q
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from ldp_action import PAR  # noqa: E402
from ldp_crossover_sweep import C_GRID, pieces  # noqa: E402

OUT = SRC.parent / "data" / "robustness"

# polynomials: coefficient lists, lowest degree first


def trim(p):
    while len(p) > 1 and p[-1] == 0:
        p = p[:-1]
    return p


def add(p, q):
    n = max(len(p), len(q))
    return trim([(p[i] if i < len(p) else 0) + (q[i] if i < len(q) else 0) for i in range(n)])


def mul(p, q):
    out = [Q(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += a * b
    return trim(out)


def deriv(p):
    return trim([i * p[i] for i in range(1, len(p))] or [Q(0)])


def divmod_poly(p, q):
    p = list(p)
    quo = [Q(0)] * max(len(p) - len(q) + 1, 1)
    while len(p) >= len(q) and any(p):
        k = len(p) - len(q)
        f = p[-1] / q[-1]
        quo[k] = f
        for i, b in enumerate(q):
            p[i + k] -= f * b
        p = trim(p[:-1]) if len(p) > 1 else p
    return trim(quo), trim(p)


def ev(p, x):
    s = Q(0)
    for a in reversed(p):
        s = s * x + a
    return s


def sturm(p):
    seq = [p, deriv(p)]
    while not (len(seq[-1]) == 1 and seq[-1][0] == 0):
        _, r = divmod_poly(seq[-2], seq[-1])
        r = [-a for a in r]
        if len(r) == 1 and r[0] == 0:
            break
        seq.append(r)
    return seq


def changes(vals):
    v = [a for a in vals if a != 0]
    return sum(1 for a, b in zip(v, v[1:]) if (a > 0) != (b > 0))


def sign_changes_at(seq, x):
    return changes([ev(p, x) for p in seq])


def sign_changes_at_inf(seq):
    return changes([p[-1] for p in seq])


def gcd_poly(p, q):
    while not (len(q) == 1 and q[0] == 0):
        _, r = divmod_poly(p, q)
        p, q = q, r
    return p


def polynomial(c):
    P = {k: Q(str(v)) for k, v in PAR.items()}
    h = int(P["h"])
    assert h == P["h"]
    s = P["aC"] / P["kappa"]
    wh = [Q(0)] * h + [s ** h]                    # (aC x / kappa)^h
    lin = [P["b0"] - P["d0"] - Q(str(c)), -P["theta"]]
    return add(mul(lin, add([P["Kh"] ** h], wh)), [P["b1"] * a for a in wh])


def isolate(seq, lo, hi, tol=Q(1, 10 ** 12)):
    """Intervals (lo, hi] each holding exactly one root, by exact bisection."""
    n = sign_changes_at(seq, lo) - sign_changes_at(seq, hi)
    if n == 0:
        return []
    if n == 1 and hi - lo < tol:
        return [(lo, hi)]
    mid = (lo + hi) / 2
    return isolate(seq, lo, mid, tol) + isolate(seq, mid, hi, tol)


def main():
    out = dict(settings=dict(par={k: str(v) for k, v in PAR.items()},
                             method="Sturm sequence, exact rationals"), costs={})
    for c in C_GRID:
        p = polynomial(c)
        seq = sturm(p)
        n_pos = sign_changes_at(seq, Q(0)) - sign_changes_at_inf(seq)
        g = gcd_poly(p, deriv(p))
        simple = len(g) == 1
        # Cauchy bound for the roots
        B = 1 + max(abs(a / p[-1]) for a in p[:-1])
        roots = [float((a + b) / 2) for a, b in isolate(seq, Q(0), B)]
        _, _, _, _, _, x_st, x_on = pieces(c)
        out["costs"][f"{c:g}"] = dict(coefficients_low_to_high=[str(a) for a in p], degree=len(p) - 1,
                                      P_at_0=str(p[0]), n_positive_roots=n_pos, all_roots_simple=simple,
                                      positive_roots=roots, float_search=[x_st, x_on])
        print(f"c={c:g}: degree {len(p) - 1}, P(0)={p[0]}, positive roots {n_pos}, simple {simple}, "
              f"roots {[f'{x:.10f}' for x in roots]} vs float search {x_st:.10f}, {x_on:.10f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sturm_equilibria.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
