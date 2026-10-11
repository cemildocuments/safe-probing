import sys, unittest, json
from pathlib import Path
import numpy as np
from numba import set_num_threads

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from engine import simulate, estimates, summarise


class ReproductionTests(unittest.TestCase):
    def test_main_prefix(self):
        rows = json.loads((ROOT / "results/main_summary.json").read_text())
        for row in rows:
            cfg = {
                k: row[k]
                for k in [
                    "mu",
                    "b",
                    "c",
                    "D",
                    "T",
                    "dt",
                    "low",
                    "high",
                    "period",
                    "duty",
                    "seed",
                ]
            }
            set_num_threads(1)
            a = simulate(32, **cfg)
            set_num_threads(min(4, __import__("numba").config.NUMBA_NUM_THREADS))
            b = simulate(32, **cfg)
            np.testing.assert_array_equal(a, b)
            old = np.load(next((ROOT / "results").glob(f"main_{row['label']}_*.npz")))[
                "statistics"
            ][:32]
            np.testing.assert_allclose(a, old, rtol=1e-12, atol=1e-12)

    def test_robust_prefix(self):
        for i, row in enumerate(
            json.loads((ROOT / "results/robust_validation.json").read_text())
        ):
            cfg = {
                k: row[k]
                for k in [
                    "mu",
                    "b",
                    "c",
                    "D",
                    "T",
                    "dt",
                    "low",
                    "high",
                    "period",
                    "duty",
                    "seed",
                    "alarm",
                ]
            }
            a = simulate(16, **cfg)
            old = np.load(
                ROOT / "results" / f"robust_validation_{row['label']}_{324+i}.npz"
            )["statistics"][:16]
            np.testing.assert_allclose(a, old, rtol=1e-12, atol=1e-12)

    def test_deterministic_contrast(self):
        from check_limits import deterministic_contrast
        from scipy.integrate import solve_ivp

        def rhs(t, y):
            x = y[0]
            return [2 - x * x, x * x, x**4]

        y = solve_ivp(rhs, [0, 20], [1, 0, 0], rtol=1e-11, atol=1e-12).y[:, -1]
        expected = 1 - y[1] ** 2 / (20 * y[2])
        actual = deterministic_contrast([(20, 1)])["C0"]
        self.assertAlmostEqual(actual, expected, places=10)

    def test_saved_summaries(self):
        for row in json.loads((ROOT / "results/main_summary.json").read_text()):
            a = np.load(next((ROOT / "results").glob(f"main_{row['label']}_*.npz")))[
                "statistics"
            ]
            r = summarise(a, row["mu"], row["D"])
            for k in ["rmse", "risk", "information", "contrast"]:
                self.assertAlmostEqual(r[k], row[k], places=11)

    def test_sharp_bound(self):
        for a, c in [(1.0, 2.0), (0.2, 10.0), (2.0, 2.5)]:
            p = c / (a + c)
            C = 1 - (p * a + (1 - p) * c) ** 2 / (p * a * a + (1 - p) * c * c)
            self.assertAlmostEqual(C, ((c - a) / (c + a)) ** 2, places=12)

    def test_estimator_normal_equations(self):
        a = np.load(next((ROOT / "results").glob("main_frontloaded_*.npz")))[
            "statistics"
        ][:100]
        e = estimates(a)
        np.testing.assert_allclose(
            a[:, 0] * e[:, 0] - a[:, 1] * e[:, 1], a[:, 5], rtol=1e-11, atol=1e-11
        )
        np.testing.assert_allclose(
            -a[:, 1] * e[:, 0] + a[:, 2] * e[:, 1], -a[:, 6], rtol=1e-11, atol=1e-11
        )

    def test_legacy_search(self):
        row = json.loads((ROOT / "results/search.json").read_text())[-1]
        cfg = {
            k: row[k]
            for k in [
                "mu",
                "b",
                "c",
                "D",
                "T",
                "dt",
                "low",
                "high",
                "period",
                "duty",
                "seed",
            ]
        }
        set_num_threads(min(4, __import__("numba").config.NUMBA_NUM_THREADS))
        a = simulate(6000, **cfg, fractional=False)
        r = summarise(a, 1.0, 0.3)
        for k in ["risk", "rmse", "information"]:
            self.assertAlmostEqual(r[k], row[k], places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
