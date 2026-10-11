"""Deterministically seeded trajectory samples for inspection and reuse."""

import numpy as np
from numba import njit, prange, set_num_threads
from pathlib import Path


@njit(parallel=True, cache=True)
def sample(n, mode, seed):
    dt = 0.000625
    stride = 32
    steps = 32000
    out = np.full((n, 1001), np.nan)
    for j in prange(n):
        np.random.seed(seed + j)
        x = 1.0
        out[j, 0] = x
        for k in range(steps):
            t = k * dt
            u = 0.0
            if mode == 1:
                u = 1.0
            if mode == 2:
                u = 6.0 if t < 20 / 11 else 0.5
            dx = (1 - x * x + u) * dt + np.sqrt(2 * 0.3 * dt) * np.random.randn()
            x += dx
            if (k + 1) % stride == 0:
                out[j, (k + 1) // stride] = x
            if x <= -1:
                break
    return out


if __name__ == "__main__":
    set_num_threads(min(12, __import__("numba").config.NUMBA_NUM_THREADS))
    p = Path(__file__).resolve().parents[1] / "results"
    data = {}
    for i, n in enumerate(["passive", "steady_dose", "frontloaded"]):
        data[n] = sample(256, i, 180000000 + i * 10000)
    np.savez_compressed(p / "sample_paths.npz", time=np.linspace(0, 20, 1001), **data)
