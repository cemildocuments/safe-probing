"""Independent absorbing Fokker--Planck check with a tridiagonal generator."""

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu
from pathlib import Path
import json


def density(
    low, high, duty=1 / 11, period=20.0, D=0.3, T=20.0, dx=0.005, dt=0.0025, right=5.0
):
    x = np.arange(-1 + dx, right, dx)
    n = len(x)
    p = np.zeros(n)
    v = (1 - x[0]) / dx
    i = int(np.floor(v))
    p[i] = 1 - (v - i)
    p[i + 1] = v - i
    sols = {}
    m = np.zeros(3)
    risk = []
    times = []

    def solver(u, h):
        k = (round(float(u), 10), h)
        if k not in sols:
            f = 1 - x * x + u
            up = D / dx**2 + f / (2 * dx)
            down = D / dx**2 - f / (2 * dx)
            up[-1] = 0.0
            assert min(up.min(), down.min()) >= 0
            Q = diags([up[:-1], -(up + down), down[1:]], [-1, 0, 1], format="csc")
            I = eye(n, format="csc")
            sols[k] = (splu(I - h / 2 * Q), I + h / 2 * Q)
        return sols[k]

    for k in range(round(T / dt)):
        phase = k % round(period / dt)
        frac = np.clip((duty * period - phase * dt) / dt, 0, 1)
        u = low + (high - low) * frac
        lu, B = solver(u, dt)
        q = lu.solve(B @ p)
        # Trapezoidal integration of unnormalised surviving density moments.
        m += (
            dt
            / 2
            * np.array([(p + q).sum(), ((p + q) * x * x).sum(), ((p + q) * x**4).sum()])
        )
        p = q
        if k % max(1, round(0.1 / dt)) == 0:
            times.append((k + 1) * dt)
            risk.append(1 - p.sum())
    return dict(
        dx=dx,
        dt=dt,
        risk=float(1 - p.sum()),
        information=float((m[0] - m[1] ** 2 / m[2]) / (2 * D)),
        mean_time=float(m[0]),
        times=times,
        cumulative_risk=risk,
    )


if __name__ == "__main__":
    out = {}
    for name, p in [
        ("passive", (0.0, 0.0)),
        ("steady_dose", (1.0, 1.0)),
        ("frontloaded", (0.5, 6.0)),
    ]:
        for dx in [0.02, 0.01, 0.005]:
            r = density(*p, dx=dx, dt=0.0025 if dx > 0.005 else 0.00125)
            out[f"{name}_{dx}"] = r
            print(name, dx, r["risk"], r["information"], flush=True)
    (Path(__file__).resolve().parents[1] / "results/density.json").write_text(
        json.dumps(out, indent=2)
    )

    if __name__ == "__main__":
        boundary = {}
        for name, p in [("passive", (0.0, 0.0)), ("frontloaded", (0.5, 6.0))]:
            r = density(*p, dx=0.005, dt=0.00125, right=7.0)
            old = out[f"{name}_0.005"]
            boundary[name] = dict(
                right=7,
                risk_difference=r["risk"] - old["risk"],
                information_difference=r["information"] - old["information"],
            )
        (
            Path(__file__).resolve().parents[1] / "results/density_boundary_check.json"
        ).write_text(json.dumps(boundary, indent=2))
