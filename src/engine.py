"""Stopped scalar diffusion experiments; fixed alarm and recorded sufficient statistics."""

import numpy as np
from numba import njit, prange

COLUMNS = [
    "time",
    "int_x2",
    "int_x4",
    "int_x6",
    "int_x8",
    "int_dy",
    "int_x2dy",
    "int_x4dy",
    "alarm",
    "dose",
    "squared_effort",
    "terminal_state",
]


@njit(parallel=True, cache=True)
def simulate(
    n,
    mu,
    b,
    c,
    D,
    T,
    dt,
    low,
    high,
    period,
    duty,
    seed,
    alarm=-1.0,
    x0=1.0,
    fractional=True,
):
    out = np.zeros((n, 12))
    steps = int(round(T / dt))
    sd = np.sqrt(2 * D * dt)
    for j in prange(n):
        np.random.seed(seed + j)
        x = x0
        for k in range(steps):
            # Integer step phase avoids floating point discontinuity ambiguity.
            phase = k % max(1, int(round(period / dt)))
            fraction = max(0.0, min(1.0, (duty * period - phase * dt) / dt))
            if not fractional:
                fraction = 1.0 if phase < int(round(duty * period / dt)) else 0.0
            u = low + (high - low) * fraction
            q = x * x
            q2 = q * q
            dx = (mu - b * q - c * q2 + u) * dt + sd * np.random.randn()
            dy = dx - u * dt
            out[j, 0] += dt
            out[j, 1] += q * dt
            out[j, 2] += q2 * dt
            out[j, 3] += q2 * q * dt
            out[j, 4] += q2 * q2 * dt
            out[j, 5] += dy
            out[j, 6] += q * dy
            out[j, 7] += q2 * dy
            out[j, 9] += u * dt
            out[j, 10] += u * u * dt
            x += dx
            if x <= alarm:
                out[j, 8] = 1.0
                break
        out[j, 11] = x
    return out


def estimates(a, c_known=0.0, fit_quartic=False):
    n = len(a)
    H = np.empty((n, 3, 3))
    H[:, 0, 0] = a[:, 0]
    H[:, 0, 1] = H[:, 1, 0] = -a[:, 1]
    H[:, 0, 2] = H[:, 2, 0] = -a[:, 2]
    H[:, 1, 1] = a[:, 2]
    H[:, 1, 2] = H[:, 2, 1] = a[:, 3]
    H[:, 2, 2] = a[:, 4]
    y = a[:, 5:8].copy()
    y[:, 1:] *= -1
    if fit_quartic:
        return np.linalg.solve(H, y)
    y2 = y[:, :2].copy()
    y2[:, 0] += c_known * a[:, 2]
    y2[:, 1] -= c_known * a[:, 3]
    return np.linalg.solve(H[:, :2, :2], y2)


def wilson(k, n, z=1.95996398454):
    p = k / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(cen - w), float(cen + w)


def summarise(a, mu, D, c_known=0.0, fit_quartic=False):
    theta = estimates(a, c_known, fit_quartic)
    e = theta[:, 0] - mu
    v = e * e
    n = len(a)
    m = a.mean(0)
    H = np.array([[m[0], -m[1], -m[2]], [-m[1], m[2], m[3]], [-m[2], m[3], m[4]]]) / (
        2 * D
    )
    if not fit_quartic:
        H = H[:2, :2]
    J = 1 / np.linalg.inv(H)[0, 0]
    rmse = np.sqrt(v.mean())
    mse_se = v.std(ddof=1) / np.sqrt(n)
    risk = m[8]
    lo, hi = wilson(int(a[:, 8].sum()), n)
    known = (a[:, 5] + a[:, 1] + c_known * a[:, 2]) / a[:, 0]
    return dict(
        n=n,
        alarms=int(a[:, 8].sum()),
        risk=float(risk),
        risk_low=lo,
        risk_high=hi,
        rmse=float(rmse),
        rmse_se=float(mse_se / (2 * rmse)),
        bias=float(e.mean()),
        information=float(J),
        known_information=float(m[0] / (2 * D)),
        contrast=float(J * 2 * D / m[0]),
        mean_time=float(m[0]),
        dose=float(m[9]),
        squared_effort=float(m[10]),
        negative_mu_fraction=float((theta[:, 0] < 0).mean()),
        negative_b_fraction=float((theta[:, 1] < 0).mean()),
    )
