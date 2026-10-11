"""Finite-horizon deterministic contrasts and passive small-noise information."""

import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad


def deterministic_contrast(segments, mu=1.0, b=1.0, x0=1.0):
    x = x0
    A2 = A4 = T = 0.0
    for length, u in segments:
        r = np.sqrt((mu + u) / b)

        def path(t):
            h = np.tanh(b * r * t)
            return r * (x + r * h) / (r + x * h)

        A2 += quad(lambda t: path(t) ** 2, 0, length, epsabs=1e-11, epsrel=1e-11)[0]
        A4 += quad(lambda t: path(t) ** 4, 0, length, epsabs=1e-11, epsrel=1e-11)[0]
        x = path(length)
        T += length
    C = 1 - A2 * A2 / (T * A4)
    return dict(T=T, A2=A2, A4=A4, C0=C, D_times_information=T * C / 2)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out = {"passive_limit": 20 - (1 - np.exp(-80)) / 4}
    out["steady_dose"] = deterministic_contrast([(20.0, 1.0)])
    out["frontloaded"] = deterministic_contrast([(20 / 11, 6.0), (200 / 11, 0.5)])
    out["robust_pulse"] = deterministic_contrast([(60 / 19, 6.0), (320 / 19, 1.25)])
    (root / "results/finite_limits.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
