"""Numerical checks of score identities, occupation bounds and small-noise rates."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from numba import njit, prange, set_num_threads
from engine import simulate

root = Path(__file__).resolve().parents[1]
set_num_threads(min(12, __import__("numba").config.NUMBA_NUM_THREADS))
checks = {}
scores = []
for p in sorted((root / "results").glob("main_*.npz")):
    a = np.load(p)["statistics"]
    D = 0.3
    s = np.column_stack([a[:, 5] - a[:, 0] + a[:, 1], -a[:, 6] + a[:, 1] - a[:, 2]]) / (
        2 * D
    )
    m = a.mean(0)
    I = np.array([[m[0], -m[1]], [-m[1], m[2]]]) / (2 * D)
    se = s.std(0, ddof=1) / len(a) ** 0.5
    z = s.mean(0) / se
    scores.append(
        dict(
            file=p.name,
            mean_score=s.mean(0).tolist(),
            score_z=z.tolist(),
            relative_covariance_error=float(
                np.linalg.norm(s.T @ s / len(a) - I) / np.linalg.norm(I)
            ),
        )
    )
checks["score_identities"] = scores
rng = np.random.default_rng(1234)
maxgap = -np.inf
for _ in range(10000):
    a = 0.1 + rng.random()
    c = a + 0.1 + 10 * rng.random()
    z = rng.uniform(a, c, 50)
    w = rng.dirichlet(np.ones(50))
    C = 1 - (w @ z) ** 2 / (w @ (z * z))
    bound = ((c - a) / (c + a)) ** 2
    maxgap = max(maxgap, C - bound)
    p = c / (a + c)
    exact = 1 - (p * a + (1 - p) * c) ** 2 / (p * a * a + (1 - p) * c * c)
    assert abs(exact - bound) < 1e-12 and C <= bound + 1e-12
checks["occupation_bound_max_violation"] = maxgap
rates = []
for D in [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]:
    for U in [0.0, 0.5, 1.0, 2.0, 4.0, 6.0]:
        r = np.sqrt(1 + U)
        sd = np.sqrt(D / (2 * r))

        def w(z):
            x = r + sd * z
            return np.exp(-((x - r) ** 2 * r + (x - r) ** 3 / 3) / D)

        norm = quad(w, -9, 9, epsabs=1e-11)[0]
        m = quad(lambda z: (r + sd * z) ** 2 * w(z), -9, 9, epsabs=1e-11)[0] / norm
        var = (
            quad(lambda z: ((r + sd * z) ** 2 - m) ** 2 * w(z), -9, 9, epsabs=1e-11)[0]
            / norm
        )
        exact = var / (var + m * m) / (2 * D)
        limit = 1 / r**3
        rates.append(dict(D=D, U=U, rate=exact, limit=limit, ratio=exact / limit))
checks["local_rates"] = rates


@njit(parallel=True)
def paired(n, seed):
    out = np.zeros((n, 3))
    dt = 0.00125
    for j in prange(n):
        np.random.seed(seed + j)
        x = 1.0
        y = 1.0
        alive = True
        for k in range(16000):
            t = k * dt
            u = 6.0 if t < 20 / 11 else 0.5
            d = (2 * 0.3 * dt) ** 0.5 * np.random.randn()
            x += (1 - x * x) * dt + d
            y += (1 - y * y + u) * dt + d
            if y < x - 1e-12:
                out[j, 0] = 1
            if x <= -1:
                out[j, 1] = 1
                break
        # Ordering is checked until the first passive alarm, exactly the coupled comparison domain.
        out[j, 2] = y - x
    return out


p = paired(20000, 160000000)
checks["coupled_paths"] = dict(
    n=20000,
    order_violations=int(p[:, 0].sum()),
    passive_alarms=int(p[:, 1].sum()),
    min_gap=float(p[:, 2].min()),
)
(root / "results/theory_checks.json").write_text(json.dumps(checks, indent=2))
print(json.dumps({k: v for k, v in checks.items() if k != "local_rates"}, indent=2))
