import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json, time
import numpy as np
from numba import set_num_threads
from engine import simulate, summarise, wilson
from pathlib import Path

set_num_threads(min(12, __import__("numba").config.NUMBA_NUM_THREADS))
root = Path(__file__).resolve().parents[1]
t = time.time()
rows = []
i = 0
for low in [0.0, 0.25, 0.5]:
    for high in [2.0, 3.0, 4.0, 6.0]:
        for period in [1.0, 2.0, 4.0, 5.0, 10.0, 20.0]:
            duty = (1 - low) / (high - low)
            cfg = dict(
                mu=1.0,
                b=1.0,
                c=0.0,
                D=0.3,
                T=20.0,
                dt=0.005,
                low=low,
                high=high,
                period=period,
                duty=duty,
                seed=12000000 + i * 10000,
            )
            a = simulate(6000, **cfg, fractional=False)
            r = summarise(a, 1, 0.3)
            r.update(cfg)
            r["one_sided_risk_upper"] = wilson(r["alarms"], 6000, 1.64485362695)[1]
            rows.append(r)
            i += 1
ok = [r for r in rows if r["one_sided_risk_upper"] <= 0.01]
best = max(ok, key=lambda r: r["information"])
(root / "results/search.json").write_text(json.dumps(rows, indent=2))
(root / "configs/selected.json").write_text(
    json.dumps({k: best[k] for k in cfg}, indent=2)
)
print("best", best)
print("elapsed", time.time() - t)
