# Safe learning near tipping points

A reproducible numerical study of learning the forcing margin of a stochastic fold while protecting against a lower-boundary crossing. The implementation combines stopped diffusion simulation, likelihood inference with an unknown restoring force, independent design and validation samples, an absorbing probability solver, and analytical checks. The release includes the generated sufficient-statistic dataset and numerical results.

## Model and principal result

The reference dynamics are `dX = (mu - b X^2 + u) dt + sqrt(2D) dW`, starting at `X = 1` with horizon `T = 20`. Observation stops at a prescribed lower alarm. Both `mu` and `b` are estimated, while `D` and the recorded input are known. Every stopped record is retained, including alarms and estimates outside the physical parameter region.

At `mu = b = 1`, `D = 0.3`, the selected input uses `u = 6` for `20/11` time units and `u = 0.5` afterwards. In 50,000 independent validation records its margin-estimation RMSE is 0.29099, compared with 0.46847 under an equal-dose constant input. The simulated alarm probability is 0.00506; an independent probability solver gives approximately 0.00569. These are results for the stated model, horizon and boundary.

A separate uncertainty-aware design uses the more conservative fixed alarm `-sqrt(0.6/1.25)`, a scheduled dose of 40, and `u = 6` for `60/19` time units followed by `u = 1.25`. It is validated across nine combinations of `mu` and `b`. The results describe the tested grid. Quartic perturbations also quantify bias from an omitted restoring-force term.

## Installation

Use Python 3.12 and an isolated environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The reference implementation uses double precision and up to 12 CPU threads. It can run with fewer threads; set `NUMBA_NUM_THREADS` before starting Python to cap the thread pool. Six CPU threads, for example:

```sh
export NUMBA_NUM_THREADS=6
```

Numerical trajectories are seeded individually and do not depend on how work is assigned to threads. The reference results were generated with NumPy 1.26.3, SciPy 1.11.4 and Numba 0.60.0. Exact equality is tested within this numerical stack; other versions and architectures may change rounding. A GPU is not required. The workflow processes one simulation condition at a time and stores compact pathwise statistics.

## Reproduce

Run the supplied validation tests first:

```sh
python -m unittest discover -s tests -v
```

The tests regenerate prefixes of the primary and uncertainty-aware validation data, compare serial and parallel execution, reconstruct reported summaries, verify likelihood normal equations and the sharp occupation bound, and compare deterministic occupation calculations against an independent ODE solver.

Run the entire experiment sequence, including searches, independent validation, sensitivities and numerical checks:

```sh
python src/run_all.py
```

Individual experiment groups can also be run:

```sh
python src/search.py
python src/run_experiments.py --group main
python src/run_experiments.py --group noise
python src/run_experiments.py --group asymptotic
python src/run_experiments.py --group convergence
python src/run_experiments.py --group robustness
python src/run_experiments.py --group quartic
python src/robust_search.py
python src/solve_density.py
python src/check_theory.py
python src/check_limits.py
python src/sample_paths.py
```

The scripts write results into `results/` and selected schedules into `configs/`. Re-running a group replaces its generated numerical output. The complete sequence contains 4,273,000 simulated records plus 20,000 coupled path pairs and 768 illustrative trajectories. The first compiled call includes Numba compilation time. A clean execution of the complete workflow regenerated all 136 numerical result files: stored arrays and parsed JSON summaries matched the distributed results exactly. All seven validation tests passed.

## Files

| File or directory | Purpose |
|---|---|
| `src/engine.py` | Parallel Euler–Maruyama trajectories, fixed-alarm stopping, sufficient statistics, two- or three-parameter likelihood estimates and error summaries. |
| `src/search.py` | Nominal 72-candidate design search with disjoint exploratory seeds. |
| `src/run_experiments.py` | Main validation, noise sensitivity, low-noise limits, step refinement, parameter sensitivity and quartic model checks. |
| `src/robust_search.py` | 36 designs across nine parameter scenarios, followed by independently seeded validation of the selected design and steady comparator. |
| `src/solve_density.py` | Independent absorbing Fokker–Planck calculation on three spatial grids. |
| `src/check_theory.py` | Score-moment checks, occupation-bound checks, local-well quadrature and coupled-path ordering. |
| `src/check_limits.py` | Deterministic occupation integrals and finite-time small-noise constants. |
| `src/sample_paths.py` | Seeded, stopped example trajectories at reduced storage frequency. |
| `src/run_all.py` | Complete execution sequence followed by tests. |
| `configs/` | Experiment inventory and the two selected schedules. |
| `results/` | Per-path sufficient statistics, trajectory samples and JSON summaries. |
| `tests/` | Reproduction and algebraic validation. |
| `requirements.txt` | Numerical dependency versions. |
| `CITATION.cff` | Machine-readable release citation. |
| `LICENSE` | MIT licence for this release. |
| `SHA256SUMS.txt` | Checksums of distributed files. |

## Data schema

Each simulation `.npz` contains a `statistics` array with one row per independent stopped record and these columns, in order:

| Index | Name | Meaning |
|---|---|---|
| 0 | `time` | Stopped observation time. |
| 1–4 | `int_x2`, `int_x4`, `int_x6`, `int_x8` | Left-point time integrals of the corresponding state powers. |
| 5 | `int_dy` | Sum of `dX - u dt`. |
| 6–7 | `int_x2dy`, `int_x4dy` | Left-point stochastic integrals of `X^2 (dX-u dt)` and `X^4 (dX-u dt)`. |
| 8 | `alarm` | One if a sampled state reached or crossed the fixed alarm; zero otherwise. |
| 9 | `dose` | Realised integral of input up to stopping. |
| 10 | `squared_effort` | Realised integral of squared input up to stopping. |
| 11 | `terminal_state` | Last sampled state, including any alarm-crossing step. |

To load one dataset:

```python
import numpy as np
from pathlib import Path
path = next(Path('results').glob('main_frontloaded_*.npz'))
a = np.load(path)['statistics']
print(a.shape, a[:, 8].mean())
```

The JSON summaries contain parameters, seed blocks, sample sizes, alarm counts and Wilson intervals, RMSE and its Monte Carlo standard error, bias, expected information, occupation contrast, duration and realised resources. `quartic_summary.json` also contains estimates obtained by omitting or jointly estimating the quartic coefficient. Fisher matrices are averaged before taking their Schur complement. RMSE is computed from every individual estimate.

`sample_paths.npz` contains `time` and one array for each of the three principal protocols; each protocol has 256 paths and trailing `NaN` values after an alarm. The first twelve paths define the displayed examples without selection by outcome. The main statistical results use the complete integration records represented by the sufficient statistics.

`density.json` contains continuous-crossing probability curves and integrated information from the independent forward solver. This solver has an absorbing boundary at -1 and a remote reflecting boundary at 5. Monte Carlo paths detect alarms on the integration grid; the convergence group quantifies the resulting monitoring error. Wilson intervals quantify sampling uncertainty, separately from this discretisation error.

## Design and reproducibility conventions

The primary and robust searches use separate seeds from all final validation. The original nominal search rounds switching times to its exploratory time grid; its `fractional=False` option preserves that exact search. Subsequent simulations average the input over the one step containing a switch, preserving scheduled dose. Squared effort in such a step follows the discretised input; exact continuous-time scheduled efforts are recorded separately in the experiment manifest.

The default alarm is -1. In the uncertainty-aware search and validation, the fixed alarm is `-sqrt(0.6/1.25)`, at or to the right of every unforced separatrix in the parameter rectangle. Initial state and alarm are kept fixed across candidate parameters when evaluating the likelihood. The nominal parameter-sensitivity group instead keeps the original -1 operational alarm. A given operational alarm is not automatically the true separatrix away from the reference parameters.

The finite-time noise limits and the local, post-transient information rate are distinct calculations. A constant input has an informative entry transient. The local-rate check concerns occupation after this transient and before escape. The code preserves the transient in every simulated record.

## Licence and citation

The release is distributed under the MIT licence. Release metadata and a software citation are provided in `CITATION.cff`.
