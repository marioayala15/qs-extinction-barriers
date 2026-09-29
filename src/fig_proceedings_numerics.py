"""Figures exposing the numerical evidence used in the proceedings paper.

The phase portrait fig_nullclines.pdf is drawn by src/basins/fig_nullclines.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Okabe-Ito colour-universal-design palette, matching the LaTeX definitions
# in main.tex so figures and tables share one colour-blind-safe scheme.
OKABE_ITO = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
)
BLUE, ORANGE, GREEN, VERMILLION = OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[2], OKABE_ITO[3]

# Colour is never the only channel separating two series: each also gets its own
# marker and dash pattern, so the panels stay readable in greyscale and under any
# colour vision.  Index these in step with OKABE_ITO.
MARKERS = ("o", "s", "^", "D", "v", "P")
DASHES = (
    "-",
    (0, (4.5, 1.6)),
    (0, (1.3, 1.3)),
    (0, (5.0, 1.4, 1.1, 1.4)),
    (0, (2.8, 1.2)),
    (0, (6.0, 1.5, 1.0, 1.5)),
)

matplotlib.rcParams.update(
    {
        "font.size": 8.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.2,
        "lines.linewidth": 1.25,
    }
)


def fit_exit(rows):
    population = np.array([row["N"] for row in rows], dtype=float)
    log_mean = np.log([row["Etau"] for row in rows])
    weight = np.array([row["n_exit"] for row in rows], dtype=float)
    design = np.column_stack([population, np.log(population), np.ones_like(population)])
    root_weight = np.sqrt(weight)
    coefficients, *_ = np.linalg.lstsq(
        root_weight[:, None] * design, root_weight * log_mean, rcond=None
    )
    residual = log_mean - design @ coefficients
    reduced_chi_square = float(weight @ residual**2) / max(len(rows) - 3, 1)
    return population, log_mean, weight, coefficients, residual, reduced_chi_square


B0, B1, KH, HILL = 0.5, 2.0, 1.0, 4
D0, THETA, ALPHA, KAPPA = 0.8, 1.0, 2.0, 1.0


def division(w):
    return B0 + B1 * w**HILL / (KH**HILL + w**HILL)


def death(x, cost):
    return D0 + cost + THETA * x


def barrier_landscape_figure(cost=0.36):
    """The eliminated-signal WKB potential as a landscape, so the barrier is a height.

    The curve is the exponent of the quasi-stationary law, Phi(x) = int_{x_on}^x log(d/b).
    On [x*, x_on] it equals the quasipotential from x_on; below x* the quasipotential
    stays at DeltaV_inf, since the descent beyond the saddle is free.
    """
    from scipy.integrate import quad
    x_on, x_star = 1.29675, 0.55881
    integrand = lambda s: np.log(death(s, cost) / division(ALPHA * s / KAPPA))
    V = np.vectorize(lambda x: quad(integrand, x_on, x)[0])

    grid = np.linspace(0.33, 1.45, 400)
    dv_inf = float(V(x_star))
    dv_r1 = 0.09526  # finite-r barrier at r = 1, from crossover_barriers.json

    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    ax.plot(grid, V(grid), color=BLUE, linewidth=1.5, zorder=3)

    ax.axhline(dv_r1, color=VERMILLION, linestyle=(0, (4, 2)), linewidth=1.1, zorder=2)
    ax.annotate(r"$\Delta V(1)$", (1.44, dv_r1), fontsize=7.5, color=VERMILLION,
                ha="right", va="bottom")

    ax.annotate("", xy=(x_star, dv_inf), xytext=(x_star, 0.0), zorder=4,
                arrowprops={"arrowstyle": "<->", "color": "0.25", "lw": 1.0})
    ax.annotate(r"$\Delta V_\infty$", (x_star, 0.5 * dv_inf), fontsize=8,
                ha="left", va="center", xytext=(5, 0), textcoords="offset points")

    ax.plot(x_on, 0.0, "o", color="0.1", markersize=5, zorder=5)
    ax.plot(x_star, dv_inf, "o", markerfacecolor="white", markeredgecolor="0.1",
            markersize=5, zorder=5)
    ax.annotate(r"$U_{\mathrm{on}}$", (x_on, 0.0), fontsize=8,
                xytext=(4, -10), textcoords="offset points")
    ax.annotate(r"$U_\ast$", (x_star, dv_inf), fontsize=8,
                xytext=(5, 3), textcoords="offset points")
    ax.annotate("extinction", xy=(0.36, 0.013), xytext=(0.50, 0.013),
                fontsize=7.5, color="0.35", va="center", ha="left",
                arrowprops={"arrowstyle": "->", "color": "0.55", "lw": 0.9})

    ax.axhline(0.0, color="0.75", linewidth=0.7, zorder=1)
    ax.set_xlabel(r"cell density $x$")
    ax.set_ylabel(r"WKB potential $\Phi_\infty(x)$")
    ax.set_xlim(0.33, 1.45)
    fig.tight_layout(pad=0.4)
    destination = ROOT / "fig_landscape.pdf"
    fig.savefig(destination)
    plt.close(fig)
    print(f"wrote {destination}")


def action_figure():
    record = json.loads((DATA / "crossover_barriers.json").read_text())
    path_record = json.loads((DATA / "optimal_paths.json").read_text())
    reference = record["costs"]["0.36"]
    rows = reference["rows"]
    r = np.array([row["r"] for row in rows])
    barrier = np.array([row["dV"] for row in rows])
    eliminated = float(reference["Vinf"])
    coefficient = float(reference["A_theory"])

    fig, axes = plt.subplots(
        1, 3, figsize=(5.7, 2.16), gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]}
    )
    colors = OKABE_ITO

    x_grid = np.linspace(
        path_record["settings"]["x_star"], path_record["settings"]["x_on"], 200
    )
    axes[0].plot(x_grid, 2.0 * x_grid, "--", color="0.45", label=r"$w=\bar w(x)$")
    for index, row in enumerate(path_record["rows"]):
        path = np.asarray(row["path"])
        axes[0].plot(path[:, 0], path[:, 1], color=colors[index],
                     linestyle=DASHES[index], label=fr"$r={row['r']:g}$")
    axes[0].plot(
        path_record["settings"]["x_on"],
        2.0 * path_record["settings"]["x_on"],
        "o",
        color="0.1",
        markersize=3.5,
    )
    axes[0].plot(
        path_record["settings"]["x_star"],
        2.0 * path_record["settings"]["x_star"],
        "o",
        markerfacecolor="white",
        markeredgecolor="0.1",
        markersize=4.0,
    )
    axes[0].set_xlabel(r"cell density $x$")
    axes[0].set_ylabel(r"signal density $w$")
    axes[0].legend(loc="upper left", fontsize=6.2, framealpha=0.92)

    rr = np.logspace(np.log10(0.45), np.log10(80.0), 300)
    axes[1].plot(rr, eliminated + coefficient / rr, color=VERMILLION, label=r"$\Delta V_\infty+A/r$")
    axes[1].axhline(eliminated, color="0.45", linestyle="--", label=r"$\Delta V_\infty$")
    axes[1].plot(r, barrier, "o", color=BLUE, markersize=3.8, label="minimum action")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"signal speed factor $r$")
    axes[1].set_ylabel(r"barrier $\Delta V$")
    axes[1].set_xlim(0.30, 110.0)
    axes[1].legend(loc="upper right", fontsize=6.2, framealpha=0.95)
    axes[1].annotate(
        "",
        xy=(1.0, barrier[1]),
        xytext=(1.0, eliminated),
        arrowprops={"arrowstyle": "<->", "color": "0.25", "lw": 1.0},
    )
    axes[1].text(
        0.90,
        0.5 * (barrier[1] + eliminated),
        "55%",
        fontsize=6.8,
        ha="right",
        va="center",
    )

    for index, cost in enumerate(record["settings"]["c_grid"]):
        cost_rows = record["costs"][str(cost)]["rows"]
        rates = np.array([row["r"] for row in cost_rows])
        plateau = np.array([row["plateau"] for row in cost_rows])
        axes[2].plot(
            rates,
            plateau,
            color=colors[index],
            marker=MARKERS[index],
            linestyle=DASHES[index],
            markersize=3.2,
        )
        axes[2].text(
            88.0,
            plateau[-1],
            fr"$c={cost:.2f}$",
            color=colors[index],
            fontsize=6.4,
            va="center",
        )
    axes[2].axvspan(4.0, 80.0, color="0.93", zorder=0)
    axes[2].set_xscale("log")
    axes[2].set_xlabel(r"signal speed factor $r$")
    axes[2].set_ylabel(r"$r[\Delta V(r)-\Delta V_\infty]$")
    axes[2].set_xlim(0.42, 700.0)

    fig.tight_layout(pad=0.6, w_pad=1.0)
    destination = ROOT / "fig_action_results.pdf"
    fig.savefig(destination)
    plt.close(fig)
    print(f"wrote {destination}")


def exit_figure():
    record = json.loads((DATA / "crossover_exit_times.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(5.7, 2.37), gridspec_kw={"width_ratios": [1.25, 1.0]})
    colors = OKABE_ITO

    for index, rate_string in enumerate(sorted(record, key=float)):
        rate = float(rate_string)
        rows = sorted(record[rate_string], key=lambda row: row["N"])
        population, log_mean, weight, coefficients, residual, reduced = fit_exit(rows)
        standard_error = 1.0 / np.sqrt(weight)
        grid = np.linspace(population.min(), population.max(), 200)
        fitted = coefficients[0] * grid + coefficients[1] * np.log(grid) + coefficients[2]
        color = colors[index]
        marker = MARKERS[index]
        axes[0].errorbar(
            population,
            log_mean,
            yerr=standard_error,
            fmt=marker,
            markersize=3.3,
            capsize=1.8,
            color=color,
        )
        axes[0].plot(grid, fitted, color=color, linestyle=DASHES[index],
                     label=fr"$r={rate:g}$")
        axes[1].errorbar(
            population,
            residual,
            yerr=standard_error,
            fmt=marker,
            markersize=3.2,
            capsize=1.8,
            color=color,
            label=fr"$r={rate:g}$",
        )

    axes[0].set_xlabel(r"population scale $N$")
    axes[0].set_ylabel(r"$\log\widehat{\mathrm{E}[\tau_N]}$")
    axes[0].legend(loc="upper left", ncol=2, framealpha=0.93)
    axes[1].axhline(0.0, color="0.4", linewidth=0.9)
    axes[1].set_xlabel(r"population scale $N$")
    axes[1].set_ylabel(r"residual in $\log\widehat{\mathrm{E}[\tau_N]}$")

    fig.tight_layout(pad=0.7, w_pad=1.5)
    destination = ROOT / "fig_exit_scaling.pdf"
    fig.savefig(destination)
    plt.close(fig)
    print(f"wrote {destination}")


if __name__ == "__main__":
    barrier_landscape_figure()
    action_figure()
    exit_figure()
