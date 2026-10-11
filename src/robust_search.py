import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json, time
from pathlib import Path
import numpy as np
from numba import set_num_threads
from engine import simulate, summarise, wilson

set_num_threads(min(12, __import__("numba").config.NUMBA_NUM_THREADS))
root = Path(__file__).resolve().parents[1]
rows = []
i = 0
t = time.time()
for low in [0.75, 1.0, 1.25, 1.5]:
    for high in [3.0, 4.0, 6.0]:
        for period in [4.0, 10.0, 20.0]:
            duty = (2 - low) / (high - low)
            sc = []
            for mu in [0.6, 1.0, 1.4]:
                for b in [0.75, 1.0, 1.25]:
                    cfg = dict(
                        mu=mu,
                        b=b,
                        c=0.0,
                        D=0.3,
                        T=20.0,
                        dt=0.005,
                        low=low,
                        high=high,
                        period=period,
                        duty=duty,
                        seed=80000000 + i * 10000,
                        alarm=-np.sqrt(0.6 / 1.25),
                    )
                    a = simulate(4000, **cfg)
                    r = summarise(a, mu, 0.3)
                    r.update(mu=mu, b=b, seed=cfg["seed"])
                    r["risk_upper_screen"] = wilson(r["alarms"], 4000, 2.539184)[1]
                    sc.append(r)
                    i += 1
            row = dict(
                low=low,
                high=high,
                period=period,
                duty=duty,
                scenarios=sc,
                worst_risk=max(x["risk_upper_screen"] for x in sc),
                worst_relative_information=min(
                    x["mu"] ** 2 * x["information"] for x in sc
                ),
            )
            rows.append(row)
best = max(
    [r for r in rows if r["worst_risk"] <= 0.01],
    key=lambda r: r["worst_relative_information"],
)
(root / "results/robust_search.json").write_text(json.dumps(rows, indent=2))
(root / "configs/robust_selected.json").write_text(json.dumps(best, indent=2))
print(
    "BEST",
    {k: v for k, v in best.items() if k != "scenarios"},
    "seconds",
    time.time() - t,
    flush=True,
)
vals = []
for mu in [0.6, 1.0, 1.4]:
    for b in [0.75, 1.0, 1.25]:
        for name, p in [
            ("robust_pulse", {k: best[k] for k in ["low", "high", "period", "duty"]}),
            ("robust_steady", dict(low=2.0, high=2.0, period=20.0, duty=1.0)),
        ]:
            cfg = (
                dict(
                    mu=mu,
                    b=b,
                    c=0.0,
                    D=0.3,
                    T=20.0,
                    dt=0.000625,
                    seed=100000000 + i * 100000,
                    alarm=-np.sqrt(0.6 / 1.25),
                )
                | p
            )
            a = simulate(40000, **cfg)
            r = summarise(a, mu, 0.3)
            r.update(cfg)
            r.update(label=name)
            r["risk_upper_bonferroni"] = wilson(r["alarms"], 40000, 2.539184)[1]
            vals.append(r)
            np.savez_compressed(
                root / "results" / f"robust_validation_{name}_{i}.npz", statistics=a
            )
            i += 1
            print(name, mu, b, r["risk"], r["rmse"], flush=True)
(root / "results/robust_validation.json").write_text(json.dumps(vals, indent=2))
