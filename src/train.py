"""Training entry point.

The first project iteration supports a random-policy smoke test. Future IPPO,
MAPPO-style, and LLM-assisted training modes should be added behind this same
config-driven entry point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.simple_spread_wrapper import SimpleSpreadWrapper, run_random_rollout
from src.utils import load_yaml, set_global_seed, write_dict_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple Spread MARL training entry point.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/random_rollout.yaml"),
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--mode",
        choices=["random"],
        default="random",
        help="Execution mode. Only random smoke test is implemented in iteration 1.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    parser.add_argument("--episodes", type=int, default=None, help="Override the config episode count.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    rollout_config = config.get("rollout", {})
    logging_config = config.get("logging", {})

    seed = int(args.seed if args.seed is not None else rollout_config.get("seed", 0))
    episodes = int(
        args.episodes if args.episodes is not None else rollout_config.get("episodes", 3)
    )
    output_path = logging_config.get("output_path", "logs/random_rollout.csv")

    set_global_seed(seed)

    wrapper = SimpleSpreadWrapper.from_dict(config.get("environment", {}))
    try:
        rows = run_random_rollout(wrapper=wrapper, episodes=episodes, seed=seed)
    finally:
        wrapper.close()

    write_dict_rows(output_path, rows)
    if rows:
        mean_return = sum(float(row["episode_return"]) for row in rows) / len(rows)
        mean_coverage = sum(float(row["coverage_distance"]) for row in rows) / len(rows)
        mean_collision_rate = sum(float(row["collision_rate"]) for row in rows) / len(rows)
    else:
        mean_return = 0.0
        mean_coverage = 0.0
        mean_collision_rate = 0.0

    print(f"Mode: {args.mode}")
    print(f"Episodes: {episodes}")
    print(f"Seed: {seed}")
    print(f"Mean episode return: {mean_return:.3f}")
    print(f"Mean coverage distance: {mean_coverage:.3f}")
    print(f"Mean collision rate: {mean_collision_rate:.3f}")
    print(f"Saved log: {output_path}")


if __name__ == "__main__":
    main()
