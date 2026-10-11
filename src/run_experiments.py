import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import json, time, platform, argparse
from pathlib import Path
import numpy as np
from numba import set_num_threads
from engine import simulate, summarise, COLUMNS

root = Path(__file__).resolve().parents[1]
set_num_threads(min(12, __import__("numba").config.NUMBA_NUM_THREADS))
base = dict(
    mu=1.0,
    b=1.0,
    c=0.0,
    D=0.3,
    T=20.0,
    dt=0.000625,
    low=0.0,
    high=0.0,
    period=20.0,
    duty=1.0,
)
policies = {
    "passive": dict(low=0.0, high=0.0),
    "steady_dose": dict(low=1.0, high=1.0),
    "periodic": dict(low=0.0, high=2.0, period=4.0, duty=0.5),
    "frontloaded": dict(low=0.5, high=6.0, period=20.0, duty=1 / 11),
    "steady_energy": dict(low=np.sqrt(3.5), high=np.sqrt(3.5)),
}


def run(label, cfg, n, seed, group, save=True):
    a = simulate(n, **cfg, seed=seed)
    row = summarise(a, cfg["mu"], cfg["D"], cfg["c"])
    row.update(cfg)
    row.update(label=label, seed=seed, group=group)
    if cfg["c"]:
        row["omitted_quartic"] = summarise(a, cfg["mu"], cfg["D"], 0.0)
        row["estimated_quartic"] = summarise(a, cfg["mu"], cfg["D"], 0.0, True)
    if save:
        np.savez_compressed(
            root / "results" / f"{group}_{label}_{seed}.npz",
            statistics=a,
            columns=np.array(COLUMNS),
        )
    print(
        group,
        label,
        "risk",
        round(row["risk"], 5),
        "rmse",
        round(row["rmse"], 5),
        "info",
        round(row["information"], 4),
        flush=True,
    )
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="main")
    args = ap.parse_args()
    rows = []
    i = 0
    t = time.time()
    if args.group == "main":
        for name, p in policies.items():
            rows.append(run(name, base | p, 50000, 30000000 + i * 100000, "main"))
            i += 1
    elif args.group == "noise":
        for D in [0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5]:
            for name in ["passive", "steady_dose", "frontloaded"]:
                rows.append(
                    run(
                        name,
                        base | policies[name] | dict(D=D, dt=0.00125),
                        15000,
                        40000000 + i * 100000,
                        "noise",
                    )
                )
                i += 1
    elif args.group == "asymptotic":
        for D in [0.0003, 0.001]:
            for name in ["passive", "steady_dose", "frontloaded"]:
                rows.append(
                    run(
                        name,
                        base | policies[name] | dict(D=D, dt=0.00125),
                        15000,
                        190000000 + i * 100000,
                        "asymptotic",
                    )
                )
                i += 1
    elif args.group == "convergence":
        for dt in [0.005, 0.0025, 0.00125, 0.0003125, 0.00015625]:
            for name in ["passive", "steady_dose", "frontloaded"]:
                rows.append(
                    run(
                        name,
                        base | policies[name] | dict(dt=dt),
                        30000,
                        50000000 + i * 100000,
                        "convergence",
                    )
                )
                i += 1
    elif args.group == "robustness":
        for mu in [0.6, 1.0, 1.4]:
            for b in [0.75, 1.0, 1.25]:
                for name in ["passive", "steady_dose", "frontloaded"]:
                    rows.append(
                        run(
                            name,
                            base | policies[name] | dict(mu=mu, b=b, dt=0.00125),
                            10000,
                            60000000 + i * 100000,
                            "robustness",
                        )
                    )
                    i += 1
    elif args.group == "quartic":
        for c in [0.02, 0.1, 0.2]:
            for D in [0.01, 0.1, 0.3]:
                for name in ["passive", "steady_dose", "frontloaded"]:
                    rows.append(
                        run(
                            name,
                            base | policies[name] | dict(b=1 - c, c=c, D=D, dt=0.00125),
                            15000,
                            70000000 + i * 100000,
                            "quartic",
                        )
                    )
                    i += 1
    (root / "results" / f"{args.group}_summary.json").write_text(
        json.dumps(rows, indent=2)
    )
    print("seconds", time.time() - t, flush=True)
