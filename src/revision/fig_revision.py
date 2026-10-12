"""Figures of the revised proceedings paper (referee points 3 and 5).

  fig_extinction_scaling.pdf  prefactor E[tau_off] exp(-N dV_num) against 1/N from the backward
                              equation, with the simulated means, and the standardized
                              differences between the two;
  fig_action_results.pdf      as in src/fig_proceedings_numerics.py, with the middle panel
                              extended to small r and the level V_0.

Reads stored data only.

    python src/revision/fig_revision.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fig_proceedings_numerics as base  # noqa: E402  (sets the Agg backend and rcParams)
import matplotlib.pyplot as plt  # noqa: E402

DATA = ROOT / "data"
REV = DATA / "revision"
RATES = (0.5, 1.0, 2.0, 4.0, 8.0)


def extinction_figure():
    analysis = json.loads((REV / "backward_equation" / "analysis.json").read_text())["rates"]
    fig, axes = plt.subplots(1, 2, figsize=(5.7, 2.37), gridspec_kw={"width_ratios": [1.25, 1.0]})
    for index, rate in enumerate(RATES):
        record = analysis[f"{rate:g}"]
        color, marker = base.OKABE_ITO[index], base.MARKERS[index]
        size = np.array(record["N"], dtype=float)
        axes[0].plot(1.0 / size, record["prefactor"], color=color, linestyle=base.DASHES[index],
                     label=fr"$r={rate:g}$")
        exact = dict(zip(record["N"], record["tau_off"]))
        rows = base.extinction_rows(rate)
        population = np.array([row["N"] for row in rows], dtype=float)
        mean = np.array([row["Etau"] for row in rows])
        relative_error = 1.0 / np.sqrt([row["n_exit"] for row in rows])
        prefactor = mean * np.exp(-population * record["dV_num"])
        axes[0].errorbar(1.0 / population, prefactor, yerr=prefactor * relative_error, fmt=marker,
                         markersize=3.0, capsize=1.6, color=color, linewidth=0.8)
        z = (np.log(mean) - np.log([exact[int(n)] for n in population])) / relative_error
        axes[1].plot(population, z, marker, markersize=3.2, color=color)
    axes[0].set_xlim(0.0, 0.0262)
    axes[0].set_xlabel(r"$1/N$")
    axes[0].set_ylabel(r"$\mathrm{E}[\tau^{\mathrm{off}}_N]\,e^{-N\Delta V_{\mathrm{num}}(r)}$")
    axes[0].legend(loc="upper left", ncol=2, framealpha=0.93)
    axes[1].axhline(0.0, color="0.4", linewidth=0.9)
    for level in (-2.0, 2.0):
        axes[1].axhline(level, color="0.6", linewidth=0.7, linestyle=":")
    axes[1].set_ylim(-3.4, 3.4)
    axes[1].set_xlabel(r"population scale $N$")
    axes[1].set_ylabel("simulation minus exact, in s.e.")
    fig.tight_layout(pad=0.7, w_pad=1.5)
    destination = ROOT / "fig_extinction_scaling.pdf"
    fig.savefig(destination)
    plt.close(fig)
    print(f"wrote {destination}")


def small_r_barriers():
    """Finest-mesh barrier at the longest horizon for each small rate."""
    rows = []
    for path in sorted((REV / "small_r").glob("r0.*.json")):
        runs = json.loads(path.read_text())
        best = min(runs, key=lambda run: (-run["T_scale"], run["dt"]))
        rows.append((best["r"], best["dV"]))
    return np.array(sorted(rows))


def action_figure():
    record = json.loads((DATA / "crossover_barriers.json").read_text())
    path_record = json.loads((DATA / "optimal_paths.json").read_text())
    reference = record["costs"]["0.36"]
    rows = reference["rows"]
    r = np.array([row["r"] for row in rows])
    barrier = np.array([row["dV"] for row in rows])
    eliminated = float(reference["Vinf"])
    coefficient = float(reference["A_theory"])
    small = small_r_barriers()
    small = small[small[:, 0] < 0.5 - 1e-9]
    slow_limit = json.loads((REV / "small_r" / "summary_extra2.json").read_text())["V0"]

    fig, axes = plt.subplots(
        1, 3, figsize=(5.7, 2.16), gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]}
    )
    colors = base.OKABE_ITO

    x_grid = np.linspace(
        path_record["settings"]["x_star"], path_record["settings"]["x_on"], 200
    )
    axes[0].plot(x_grid, 2.0 * x_grid, "--", color="0.45", label=r"$w=\bar w(x)$")
    for index, row in enumerate(path_record["rows"]):
        path = np.asarray(row["path"])
        axes[0].plot(path[:, 0], path[:, 1], color=colors[index],
                     linestyle=base.DASHES[index], label=fr"$r={row['r']:g}$")
    axes[0].plot(path_record["settings"]["x_on"], 2.0 * path_record["settings"]["x_on"],
                 "o", color="0.1", markersize=3.5)
    axes[0].plot(path_record["settings"]["x_star"], 2.0 * path_record["settings"]["x_star"],
                 "o", markerfacecolor="white", markeredgecolor="0.1", markersize=4.0)
    axes[0].set_xlabel(r"cell density $x$")
    axes[0].set_ylabel(r"signal density $w$")
    axes[0].legend(loc="upper left", fontsize=6.2, framealpha=0.92)

    rr = np.logspace(np.log10(0.45), np.log10(80.0), 300)
    axes[1].plot(rr, eliminated + coefficient / rr, color=base.VERMILLION,
                 label=r"$\Delta V_\infty+A/r$")
    axes[1].axhline(eliminated, color="0.45", linestyle="--", label=r"$\Delta V_\infty$")
    axes[1].axhline(slow_limit, color="0.45", linestyle=":", label=r"$V_0$")
    axes[1].plot(r, barrier, "o", color=base.BLUE, markersize=3.8, label="minimum action")
    axes[1].plot(small[:, 0], small[:, 1], "o", color=base.BLUE, markersize=3.8,
                 markerfacecolor="white")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"signal speed factor $r$")
    axes[1].set_ylabel(r"barrier $\Delta V$")
    axes[1].set_xlim(0.005, 110.0)
    axes[1].set_ylim(0.04, 0.30)
    axes[1].text(70.0, slow_limit - 0.006, r"$V_0$", fontsize=6.4, ha="right", va="top")
    axes[1].text(0.007, eliminated + 0.005, r"$\Delta V_\infty$", fontsize=6.4, va="bottom")
    axes[1].text(1.6, 0.105, r"$\Delta V_\infty+A/r$", fontsize=6.4, color=base.VERMILLION,
                 va="bottom")

    for index, cost in enumerate(record["settings"]["c_grid"]):
        cost_rows = record["costs"][str(cost)]["rows"]
        rates = np.array([row["r"] for row in cost_rows])
        plateau = np.array([row["plateau"] for row in cost_rows])
        axes[2].plot(rates, plateau, color=colors[index], marker=base.MARKERS[index],
                     linestyle=base.DASHES[index], markersize=3.2)
        axes[2].text(88.0, plateau[-1], fr"$c={cost:.2f}$", color=colors[index],
                     fontsize=6.4, va="center")
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


if __name__ == "__main__":
    extinction_figure()
    action_figure()
