# Extinction barriers in a quorum-sensing reaction network

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23062341.svg)](https://doi.org/10.5281/zenodo.23062341)

Code and data for

> M. Ayala and J. Zimmer, *Computing Extinction Barriers in a Quorum-Sensing Reaction
> Network*.

A four-channel cell-signal reaction network couples a cell density `x` to a signal
density `w` through cell division, cell death, signal production and signal removal.
Multiplying signal production and removal by a common factor `r` leaves every
deterministic equilibrium and its stability type unchanged. The rare-event barrier
nevertheless depends on `r`, and this repository computes it.

The extinction time is the first arrival of the process in a fixed neighbourhood
`R_off = [0, 0.3] x [0, 0.7]` of the extinction state, which lies in its deterministic
basin. Entry into `R_off` is not yet cell extinction, `X = 0`, and recovery remains
possible. The quasipotential barrier `DeltaV(r)` is the least action from the cooperative
state `U_on` to the saddle `U_*`; by a standard Freidlin-Wentzell argument, which the
paper states without proof, it is also the least interior action needed to reach
`R_off`. It is computed and tested in three ways:

1. minimization of the Freidlin-Wentzell action, with the signal kept as a fluctuating
   coordinate rather than slaved to `wbar(x) = aC x / kappa`;
2. exact stochastic simulation of the first time the cell density reaches the threshold
   `x*`, with right-censored times regressed in the population scale `N`;
3. exact stochastic simulation of the extinction time itself, i.e. the first arrival in
   `R_off`, which includes every failed attempt.

The threshold time of route 2 has its own barrier `DeltaV_thr(r) <= DeltaV(r)`. At the
working cost `c = 0.36` the two coincide for `r >= 1`, but at `r = 0.5` a threshold point
with high signal is 7.7 percent cheaper than the saddle. The extinction-time slopes of
route 3 lie within two standard errors of `DeltaV(r)` at every simulated rate.

## The numbers

Eliminating the signal first gives a barrier available in closed form as a
one-dimensional quadrature:

    DeltaV_inf = 0.0613795 .

Keeping the signal explicit gives, at the production settings below:

    r      0.5        1         2         4         8         16        32        64
    dV     0.122477   0.095259  0.079057  0.070313  0.065843  0.063605  0.062490  0.061935

So `DeltaV(1)` exceeds `DeltaV_inf` by 55 percent, `DeltaV(r)` falls by a factor near
1.98 across the rate grid, and `r (DeltaV(r) - DeltaV_inf)` settles at `0.03553`,
against the predicted coefficient `A = 0.03544`. Under the metastable exit estimates
discussed in the paper, the mean extinction time grows as `exp(N DeltaV(r))`, so these
differences are amplified exponentially in the population scale.

At `c = 0.36` and the rates of the stochastic campaigns (slopes with their regression
errors in the last digits):

    r                        0.5          1            2            4            8
    threshold barrier        0.1131       0.0953       0.0791       0.0703       0.0658
    threshold-time slope     0.1194(102)  0.1092(95)   0.0816(88)   0.0806(57)   0.0615(46)
    extinction-time slope    0.1294(35)   0.0994(35)   0.0837(50)   0.0702(43)   0.0697(37)

Started at the saddle, the probability of reaching `R_off` before returning near `U_on`
falls from 0.63-0.64 at `N = 50` to 0.51-0.52 at `N = 3200`, at every rate.

## Layout

    src/ldp_action.py            reaction network, Lagrangian by Newton, minimum action
    src/ldp_crossover_sweep.py   the barrier sweep over (c, r); writes crossover_barriers.json
    src/ldp_crossover_exit.py    SSA with censored threshold times; writes crossover_exit_times.json
    src/paths.py                 where stored measurements go
    src/generate_optimal_paths.py   representative minimum-action paths for the landscape figure
    src/verify_hessian.py        the action solver against the Gaussian regime
    src/verify_exit_multiseed.py five-seed diagnostics for the exit-time estimator
    src/fig_proceedings_numerics.py  three figures of the paper (landscape, action, exit times)

    src/basins/                  equilibria at all costs, separatrices, threshold-line
                                 classification, and the phase portrait fig_nullclines.pdf
    src/threshold_endpoint/      the threshold barrier: profile over the terminal signal,
                                 refinement, verification, and an independent solver
    src/collapse_time/           C simulator recording threshold and extinction times along each
                                 trajectory, campaign runners, validation and analysis
    src/committor/               C simulator and runner for the success probability near the saddle
    src/robustness/              checks added in revision: regression models, lattice rounding,
                                 bootstrap, exponentiality, multi-start minimization, exact
                                 equilibrium count

    data/                        stored output of the runs that are too slow to repeat;
                                 data/basins, data/threshold_endpoint, data/collapse_time,
                                 data/committor and data/robustness hold the output of the
                                 five folders above

Everything in `data/` is regenerable by the scripts above. It is tracked because the
stochastic sweeps take hours and the figures should not require them. Two kinds of file
are not tracked because they regenerate in seconds to minutes: the manifold traces
`data/basins/*.npy` and the per-trajectory committor samples `data/committor/raw/`.

## Requirements

Python 3.9 or later, with `numpy`, `scipy` and `matplotlib`, and a C compiler for the two
simulators. The results quoted here were produced with Python 3.9.6, numpy 1.26.3, scipy
1.12.0 and matplotlib 3.8.2, except for `src/basins/` and
`src/threshold_endpoint/independent_check/`, which were run with Python 3.13.1, numpy
2.5.2, scipy 1.18.0 and matplotlib 3.11.1. With the older environment,
`fig_nullclines.pdf` has the same content but slightly different legend and line
rendering, and `data/basins/results.json` differs only in ODE entry times (relative
differences below 1e-4) and timings; every classification is unchanged.

The figure files in the submitted paper were rendered with matplotlib 3.11.1, except
`fig_exit_scaling.pdf` (3.8.2). In the revision, `fig_landscape.pdf` (relabelled as the
WKB potential) and the new `fig_extinction_scaling.pdf` (the extinction-time fits, which
replace `fig_exit_scaling.pdf` in the paper) were rendered with matplotlib 3.11.1. With matplotlib 3.8.2, `fig_landscape.pdf` and
`fig_action_results.pdf` have the same curves, colours and labels, with slightly different
tick placement and text rendering.

Compile the simulators once:

    cc -O3 -march=native -o src/collapse_time/ssa_offtarget src/collapse_time/ssa_offtarget.c -lm
    cc -O3 -march=native -o src/committor/ssa_commit src/committor/ssa_commit.c -lm

## Reproducing the paper

The figures read only stored data and take a few seconds:

    python src/fig_proceedings_numerics.py      # fig_landscape, fig_action_results, fig_exit_scaling,
                                                # fig_extinction_scaling
    cd src/basins && python separatrix.py && python fig_nullclines.py   # fig_nullclines, about 20 s

These write the PDFs to the repository root.

Three checks are cheap enough to run directly. The first calibrates the action solver
against the closed-form quadrature in the eliminated-signal model, where the answer is
known:

    python src/ldp_action.py                    # agrees to 0.07 percent at c = 0.36

This self-test is quick and coarse (500 nodes, horizons up to 80, the default iteration
budget). At the production settings `T = 40`, `M = 2000` and an outer budget of 60000
iterations, the same solver gives `0.0613801` against the quadrature value `0.0613795`,
i.e. about 0.001 percent, which is the calibration quoted in the paper:

    cd src && python -c "import ldp_action as L; print(L.mam_action(L.net_1d_slaved(0.36), [1.2967495269504323], [0.5588112173244356], T=40.0, K=2000, maxiter=60000)['S'])"

The second checks the solver against the Gaussian regime near the stable state, where
the quasipotential Hessian must satisfy `H J + J' H + H D H = 0` and is therefore the
inverse of the stationary covariance of the central limit regime. This shares nothing
with the first check:

    python src/verify_hessian.py                # worst relative error 2.5e-04

The third checks the horizon and the mesh separately, i.e. horizon convergence at fixed
step and mesh convergence at fixed horizon:

    python src/ldp_crossover_sweep.py --check

The production sweep itself is `T = 40`, `M = 2000` and an outer budget of 60000
iterations, over `r` in `{0.5, 1, 2, 4, 8, 16, 32, 64}` and `c` in
`{0.20, 0.30, 0.36, 0.45, 0.50}`:

    python src/ldp_crossover_sweep.py --write   # 40 minimizations, about four minutes

Notice that at the default 600 iterations L-BFGS stops on the cap and returns
`0.128412` at `r = 0.5` instead of `0.122477`, i.e. the budget itself moves the answer
upward by five percent, which looks exactly like a discretisation error and is not one.
Rerunning the sweep reproduces the stored barriers to about 4e-8, not bit for bit.

The threshold-time route is the expensive one. Each pair `(r, N)` gets its own random
stream, seeded reproducibly from the triple `(20260812, 100r, N)`:

    python src/ldp_crossover_exit.py --quick    # a reduced grid, for a sanity check
    python src/ldp_crossover_exit.py --write    # the full sweep, hours
    python src/ldp_crossover_exit.py --verify   # estimator diagnostics at one point

    python src/verify_exit_multiseed.py         # five seeds at r = 0.5, N = 60

### The basins and the threshold line

    cd src/basins
    python separatrix.py         # equilibria at five costs, R_off conditions, separatrices,
                                 # classification
    python action_path_r8.py     # the r = 8 minimum-action path drawn in fig_nullclines
    python tables.py             # prints the tables of data/basins/results.json

### The threshold barrier

    cd src/threshold_endpoint
    python run_profile.py        # the profile w -> V((x*, w)), about 30 minutes
    python run_refine.py         # refinement near the minima
    python run_verify.py         # horizon, mesh and seed checks of the r = 0.5 candidate
    python plot_profile.py       # fig_threshold_profile.pdf
    cd independent_check && python indep_mam.py 0.5 && python detour.py

`independent_check/` is a separately written solver (explicit Lagrangian, analytic
gradient) that shares no code with `ldp_action.py`. It reproduces the r = 0.5 threshold
and saddle actions, 0.1130794 and 0.1224781, and gives 0.033455 for the action from the
threshold candidate to the saddle.

### The extinction time

    cd src/collapse_time
    python validate.py           # simulator validation against the stored threshold data
    python run_pilot.py          # pilot grid (r = 0.5, 1, 8)
    python run_full.py           # full grid; points already in data/collapse_time/raw are skipped
    python analyze.py            # fits, bootstrap, recovery statistics, fig_offtarget_full.pdf

The full grid took about 1.3e5 core-seconds. Its horizon is `30 exp(N DeltaV_num)`,
since the mean extinction time is 3 to 4 times `exp(N DeltaV_num)` at these sizes. From the
stored raw data, `analyze.py` reproduces `fit_results.json` exactly.

### The success probability near the saddle

    cd src/committor
    python run_committor.py --workers 10 --nrep 20000 --nchunk 20   # about 4 minutes

This reproduces `data/committor/results.json` exactly, apart from timings.

### Robustness checks

    cd src/robustness
    python stats_robustness.py      # fits with and without log N, dropped sizes, the lattice
                                    # regressor frac(N x*) with its null, exponentiality; seconds
    python bootstrap_threshold.py   # parametric bootstrap of every slope; about a minute
    python lattice_threshold.py     # stop at floor(N x*) - k, k = -1..3; about 24 minutes on 10 cores
    python mam_robustness.py        # four initial paths at five rates, max|H|, exact
                                    # piecewise-linear action; about 10 minutes
    python sturm_equilibria.py      # exact count of positive equilibria at five costs

These read the stored data of the earlier folders and write `data/robustness/`. The
threshold times of the main campaign are stored as per-size summaries only, hence their
bootstrap is parametric; on the extinction campaign it agrees with the bootstrap over
trajectories to 0.0003. The quick run of the lattice test (`--nrep 200`) takes seconds.

## What this does not settle

The barrier is computed for one network at one working cost, and the stochastic routes
cover `r` in `{0.5, 1, 2, 4, 8}` only, since the horizon needed at larger `r` grows with
the barrier. The prefactor `C(N)` in `E[tau_N] = C(N) exp(N DeltaV)` is not computed: it
is absorbed into a regression on `N` and `log N`, with the uncertainty scaled to unit
reduced chi-square, and `src/ldp_crossover_exit.py --verify` is what licenses that
scaling. The relation between the action barrier and the mean extinction time relies on
metastable exit estimates for the jump process that the paper states but does not prove;
the stochastic results support it over the simulated sizes only. The identification of the
saddle barrier with the least action needed to reach `R_off` is likewise stated, not
proved. The minimizer is local, and no claim is made that it is globally reliable in
higher dimensions.

## Citing

Please cite the paper above. To cite this code and data, use the archived release on
Zenodo: version `v1.0-mmsc2026`, the one the paper cites, is
[doi:10.5281/zenodo.23062342](https://doi.org/10.5281/zenodo.23062342), and
[doi:10.5281/zenodo.23062341](https://doi.org/10.5281/zenodo.23062341) always resolves to
the latest version. `CITATION.cff` has the metadata; GitHub shows it under "Cite this
repository".

## Licence

The code is released under the MIT licence (see `LICENSE`). The data in `data/` and the
figure files are released under the Creative Commons Attribution 4.0 International
licence (CC BY 4.0).

## Use of AI assistance

The code, the numerical checks and this README were developed with the help of Claude
(Anthropic), an AI assistant, working with the authors. It wrote and tested much of the
code, ran the calibration and validation checks, and cross-checked the numbers reported in
the paper against the stored data. The authors specified the computations, reviewed the
code and results, and take full responsibility for them.
