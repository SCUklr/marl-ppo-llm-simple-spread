"""PettingZoo Simple Spread wrapper for MARL smoke tests and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from pettingzoo.mpe import simple_spread_v3


@dataclass(frozen=True)
class SimpleSpreadConfig:
    """Configuration for the PettingZoo Simple Spread environment."""

    num_agents: int = 3
    num_landmarks: int = 3
    max_cycles: int = 25
    continuous_actions: bool = True
    local_ratio: float = 0.5
    render_mode: str | None = None


class SimpleSpreadWrapper:
    """Thin wrapper around PettingZoo parallel Simple Spread.

    The wrapper keeps the first iteration intentionally small: it supports
    environment reset/step, random action sampling, global observation creation,
    and basic cooperation metrics for smoke tests.
    """

    def __init__(self, config: SimpleSpreadConfig | None = None) -> None:
        self.config = config or SimpleSpreadConfig()
        self.env = simple_spread_v3.parallel_env(
            N=self.config.num_agents,
            local_ratio=self.config.local_ratio,
            max_cycles=self.config.max_cycles,
            continuous_actions=self.config.continuous_actions,
            render_mode=self.config.render_mode,
        )
        self.possible_agents = list(self.env.possible_agents)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "SimpleSpreadWrapper":
        """Build the wrapper from a YAML environment section."""
        return cls(
            SimpleSpreadConfig(
                num_agents=int(config.get("num_agents", 3)),
                num_landmarks=int(config.get("num_landmarks", 3)),
                max_cycles=int(config.get("max_cycles", 25)),
                continuous_actions=bool(config.get("continuous_actions", True)),
                local_ratio=float(config.get("local_ratio", 0.5)),
                render_mode=config.get("render_mode"),
            )
        )

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment and return observations plus infos."""
        observations, infos = self.env.reset(seed=seed)
        return observations, infos

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        """Step the parallel environment."""
        return self.env.step(actions)

    def sample_actions(self) -> dict[str, np.ndarray]:
        """Sample one random action for every active agent."""
        return {
            agent: self.env.action_space(agent).sample()
            for agent in self.env.agents
        }

    def observation_dim(self) -> int:
        """Return the local observation dimension."""
        return int(np.prod(self.env.observation_space(self.possible_agents[0]).shape))

    def action_dim(self) -> int:
        """Return the action dimension."""
        return int(np.prod(self.env.action_space(self.possible_agents[0]).shape))

    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return low and high action bounds for continuous actions."""
        space = self.env.action_space(self.possible_agents[0])
        low = np.asarray(space.low, dtype=np.float32)
        high = np.asarray(space.high, dtype=np.float32)
        return low, high

    def global_observation_dim(self) -> int:
        """Return the concatenated observation dimension."""
        return self.observation_dim() * len(self.possible_agents)

    def global_observation(self, observations: dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate agent observations into a simple centralized input."""
        ordered = [observations[agent] for agent in self.possible_agents if agent in observations]
        if not ordered:
            return np.array([], dtype=np.float32)
        return np.concatenate(ordered).astype(np.float32)

    def cooperation_metrics(self) -> dict[str, float]:
        """Compute coverage distance and collision count from MPE world state."""
        world = getattr(getattr(self.env, "unwrapped", self.env), "world", None)
        if world is None:
            return {"coverage_distance": float("nan"), "collision_count": float("nan")}

        agent_positions = np.array([agent.state.p_pos for agent in world.agents], dtype=np.float32)
        landmark_positions = np.array(
            [landmark.state.p_pos for landmark in world.landmarks],
            dtype=np.float32,
        )

        if len(agent_positions) == 0 or len(landmark_positions) == 0:
            coverage_distance = float("nan")
        else:
            distances = np.linalg.norm(
                landmark_positions[:, None, :] - agent_positions[None, :, :],
                axis=-1,
            )
            coverage_distance = float(np.mean(np.min(distances, axis=1)))

        collision_count = 0
        for first, second in combinations(world.agents, 2):
            first_pos = np.asarray(first.state.p_pos)
            second_pos = np.asarray(second.state.p_pos)
            min_distance = float(getattr(first, "size", 0.0) + getattr(second, "size", 0.0))
            if np.linalg.norm(first_pos - second_pos) < min_distance:
                collision_count += 1

        return {
            "coverage_distance": coverage_distance,
            "collision_count": float(collision_count),
        }

    def state_snapshot(self) -> dict[str, dict[str, np.ndarray]]:
        """Return current agent and landmark positions for guidance modules."""
        world = getattr(getattr(self.env, "unwrapped", self.env), "world", None)
        if world is None:
            return {"agents": {}, "landmarks": {}}

        agents = {
            agent.name: np.asarray(agent.state.p_pos, dtype=np.float32)
            for agent in world.agents
        }
        landmarks = {
            f"landmark_{idx}": np.asarray(landmark.state.p_pos, dtype=np.float32)
            for idx, landmark in enumerate(world.landmarks)
        }
        return {"agents": agents, "landmarks": landmarks}

    def close(self) -> None:
        """Close the underlying environment."""
        self.env.close()


def run_random_rollout(
    wrapper: SimpleSpreadWrapper,
    episodes: int,
    seed: int,
) -> list[dict[str, float | int]]:
    """Run random-policy episodes and return episode-level metrics."""
    rows: list[dict[str, float | int]] = []

    for episode in range(episodes):
        observations, _ = wrapper.reset(seed=seed + episode)
        done = False
        step_count = 0
        total_return = 0.0
        collision_sum = 0.0
        final_coverage = float("nan")

        while not done and wrapper.env.agents:
            _ = wrapper.global_observation(observations)
            observations, rewards, terminations, truncations, _ = wrapper.step(wrapper.sample_actions())
            metrics = wrapper.cooperation_metrics()

            total_return += float(sum(rewards.values()))
            collision_sum += float(metrics["collision_count"])
            final_coverage = float(metrics["coverage_distance"])
            step_count += 1
            done = all(terminations.values()) or all(truncations.values())

        rows.append(
            {
                "episode": episode,
                "seed": seed + episode,
                "steps": step_count,
                "episode_return": total_return,
                "coverage_distance": final_coverage,
                "collision_rate": collision_sum / max(step_count, 1),
            }
        )

    return rows
