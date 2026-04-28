"""Run the lightweight 3-method x 3-seed experiment suite."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_RUNS = [
    ("ippo", "configs/ippo.yaml"),
    ("mappo", "configs/mappo.yaml"),
    ("llm_guidance", "configs/llm_guidance.yaml"),
]


def default_python_path() -> Path:
    """Use the active venv when present, otherwise choose a platform default."""
    if os.environ.get("VIRTUAL_ENV"):
        return Path(sys.executable)
    if os.name == "nt":
        return Path(".venv/Scripts/python.exe")
    return Path(".venv/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight Simple Spread experiments.")
    parser.add_argument("--episodes", type=int, default=500, help="Episodes per method/seed run.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seeds to run.")
    parser.add_argument(
        "--python",
        type=Path,
        default=default_python_path(),
        help="Python executable to use.",
    )
    parser.add_argument(
        "--skip_plot",
        action="store_true",
        help="Skip result plotting after training.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    python = str(args.python)

    for method, config in DEFAULT_RUNS:
        for seed in args.seeds:
            print(f"\n=== Running {method} seed={seed} episodes={args.episodes} ===", flush=True)
            run_command(
                [
                    python,
                    "src/train.py",
                    "--config",
                    config,
                    "--episodes",
                    str(args.episodes),
                    "--seed",
                    str(seed),
                ]
            )

    if not args.skip_plot:
        run_command([python, "src/plot_results.py", "--output_dir", "results"])


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
