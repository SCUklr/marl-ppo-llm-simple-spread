"""Evaluate saved Simple Spread policies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.common import GaussianActor, device_from_config, device_summary, tensor
from src.envs.simple_spread_wrapper import SimpleSpreadWrapper
from src.utils import load_yaml, write_dict_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Simple Spread policy.")
    parser.add_argument("--config", type=Path, required=True, help="Config used for training.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=10_000, help="Evaluation seed.")
    parser.add_argument("--output", type=Path, default=Path("results/evaluation.csv"))
    return parser.parse_args()


def build_actor(config: dict, env: SimpleSpreadWrapper, device: torch.device) -> GaussianActor:
    algo = config.get("algorithm", {})
    hidden_sizes = [int(size) for size in algo.get("hidden_sizes", [64, 64])]
    action_low, action_high = env.action_bounds()
    return GaussianActor(
        obs_dim=env.observation_dim(),
        action_dim=env.action_dim(),
        hidden_sizes=hidden_sizes,
        action_low=action_low,
        action_high=action_high,
        log_std_init=float(algo.get("log_std_init", -0.5)),
    ).to(device)


def evaluate(config: dict, checkpoint_path: Path, episodes: int, seed: int) -> list[dict[str, float | int | str]]:
    env = SimpleSpreadWrapper.from_dict(config.get("environment", {}))
    device = device_from_config(config.get("training", {}))
    runtime = device_summary(device)
    actor = build_actor(config, env, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    rows: list[dict[str, float | int | str]] = []
    method = str(checkpoint.get("method", config.get("algorithm", {}).get("name", "unknown")))

    try:
        for episode in range(episodes):
            observations, _ = env.reset(seed=seed + episode)
            done = False
            step_count = 0
            episode_return = 0.0
            collision_sum = 0.0
            final_coverage = float("nan")

            while not done and env.env.agents:
                actions: dict[str, np.ndarray] = {}
                for agent in env.env.agents:
                    obs = np.asarray(observations[agent], dtype=np.float32)
                    with torch.no_grad():
                        action = actor.deterministic_action(tensor(obs[None, :], device))
                    actions[agent] = action.squeeze(0).cpu().numpy().astype(np.float32)

                observations, rewards, terminations, truncations, _ = env.step(actions)
                metrics = env.cooperation_metrics()
                done = all(terminations.values()) or all(truncations.values())

                episode_return += float(sum(rewards.values()))
                collision_sum += float(metrics["collision_count"])
                final_coverage = float(metrics["coverage_distance"])
                step_count += 1

            rows.append(
                {
                    "method": method,
                    "episode": episode,
                    "seed": seed + episode,
                    "steps": step_count,
                    "episode_return": episode_return,
                    "coverage_distance": final_coverage,
                    "collision_rate": collision_sum / max(step_count, 1),
                    "device": runtime["device"],
                    "device_name": runtime["device_name"],
                }
            )
    finally:
        env.close()

    return rows


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    runtime = device_summary(device_from_config(config.get("training", {})))
    rows = evaluate(config, args.checkpoint, args.episodes, args.seed)
    write_dict_rows(args.output, rows)
    mean_return = float(np.mean([row["episode_return"] for row in rows]))
    mean_coverage = float(np.mean([row["coverage_distance"] for row in rows]))
    mean_collision = float(np.mean([row["collision_rate"] for row in rows]))
    print(f"Evaluated {args.checkpoint}")
    print(
        f"Runtime device: {runtime['device']} ({runtime['device_name']}) "
        f"cuda_available={runtime['cuda_available']}"
    )
    print(f"Mean return: {mean_return:.3f}")
    print(f"Mean coverage distance: {mean_coverage:.3f}")
    print(f"Mean collision rate: {mean_collision:.3f}")
    print(f"Saved evaluation: {args.output}")


if __name__ == "__main__":
    main()
