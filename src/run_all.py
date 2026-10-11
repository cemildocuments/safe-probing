"""Run the complete deterministic-seed numerical workflow."""

from pathlib import Path
import subprocess, sys

root = Path(__file__).resolve().parents[1]
commands = (
    [["search.py"]]
    + [
        ["run_experiments.py", "--group", g]
        for g in ["main", "noise", "asymptotic", "convergence", "robustness", "quartic"]
    ]
    + [
        [s]
        for s in [
            "robust_search.py",
            "solve_density.py",
            "check_theory.py",
            "check_limits.py",
            "sample_paths.py",
        ]
    ]
)
for command in commands:
    print("Running", " ".join(command), flush=True)
    subprocess.run(
        [sys.executable, str(root / "src" / command[0]), *command[1:]],
        cwd=root,
        check=True,
    )
subprocess.run(
    [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
    cwd=root,
    check=True,
)
