"""Run short smoke tests for all supported experiment modes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def default_python_path() -> Path:
    if os.environ.get("VIRTUAL_ENV"):
        return Path(sys.executable)
    if os.name == "nt":
        return Path(".venv/Scripts/python.exe")
    return Path(".venv/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run smoke tests for Simple Spread experiments.")
    parser.add_argument("--episodes", type=int, default=4, help="Episodes per smoke test.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for smoke tests.")
    parser.add_argument(
        "--python",
        type=Path,
        default=default_python_path(),
        help="Python executable to use.",
    )
    parser.add_argument(
        "--skip_llm_provider_test",
        action="store_true",
        help="Skip the standalone LLM provider smoke test.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    python = str(args.python)
    smoke_configs = [
        "configs/random_rollout.yaml",
        "configs/ippo.yaml",
        "configs/mappo.yaml",
        "configs/llm_guidance.yaml",
    ]

    run_command([python, "scripts/verify_runtime.py"])
    if not args.skip_llm_provider_test:
        run_command([python, "scripts/test_llm_provider.py", "--config", "configs/llm_guidance.yaml"])

    for config in smoke_configs:
        run_command(
            [
                python,
                "src/train.py",
                "--config",
                config,
                "--episodes",
                str(args.episodes),
                "--seed",
                str(args.seed),
            ]
        )


if __name__ == "__main__":
    main()
