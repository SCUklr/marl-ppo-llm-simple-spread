"""Independent PPO baseline for PettingZoo Simple Spread."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.algorithms.common import (
    GaussianActor,
    RolloutBatch,
    ValueCritic,
    compute_gae,
    device_from_config,
    normalize,
    tensor,
)
from src.envs.simple_spread_wrapper import SimpleSpreadWrapper
from src.llm.guidance import GuidanceDecision, GuidanceProvider
from src.utils import append_dict_rows, ensure_parent_dir, set_global_seed


class IPPOTrainer:
    """Shared-parameter Independent PPO trainer."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.env = SimpleSpreadWrapper.from_dict(config.get("environment", {}))
        algo = config.get("algorithm", {})
        training = config.get("training", {})

        self.seed = int(training.get("seed", 0))
        set_global_seed(self.seed)
        self.device = device_from_config(training)

        obs_dim = self.env.observation_dim()
        action_dim = self.env.action_dim()
        action_low, action_high = self.env.action_bounds()
        hidden_sizes = [int(size) for size in algo.get("hidden_sizes", [64, 64])]

        self.actor = GaussianActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            action_low=action_low,
            action_high=action_high,
            log_std_init=float(algo.get("log_std_init", -0.5)),
        ).to(self.device)
        self.critic = ValueCritic(obs_dim, hidden_sizes).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=float(algo.get("learning_rate", 3e-4)),
        )

        self.gamma = float(algo.get("gamma", 0.99))
        self.gae_lambda = float(algo.get("gae_lambda", 0.95))
        self.clip_coef = float(algo.get("clip_coef", 0.2))
        self.value_coef = float(algo.get("value_coef", 0.5))
        self.entropy_coef = float(algo.get("entropy_coef", 0.01))
        self.max_grad_norm = float(algo.get("max_grad_norm", 0.5))
        self.update_epochs = int(algo.get("update_epochs", 4))
        self.minibatch_size = int(algo.get("minibatch_size", 256))

        self.total_episodes = int(training.get("total_episodes", 100))
        self.episodes_per_update = int(training.get("episodes_per_update", 4))
        self.log_interval = int(training.get("log_interval", 1))

        logging = config.get("logging", {})
        self.method = str(algo.get("name", "ippo"))
        path_values = {"seed": self.seed, "method": self.method}
        self.log_path = str(logging.get("log_path", "logs/ippo_seed_{seed}.csv")).format(
            **path_values
        )
        self.checkpoint_path = str(
            logging.get("checkpoint_path", "checkpoints/ippo_seed_{seed}_best.pt")
        ).format(**path_values)
        self.guidance = GuidanceProvider(config.get("llm_guidance", {}))

    def collect_rollouts(self, start_episode: int) -> tuple[RolloutBatch, list[dict[str, float | int]]]:
        """Collect several full episodes and flatten per-agent samples."""
        obs_by_agent: dict[str, list[np.ndarray]] = {agent: [] for agent in self.env.possible_agents}
        actions_by_agent: dict[str, list[np.ndarray]] = {agent: [] for agent in self.env.possible_agents}
        log_probs_by_agent: dict[str, list[float]] = {agent: [] for agent in self.env.possible_agents}
        rewards_by_agent: dict[str, list[float]] = {agent: [] for agent in self.env.possible_agents}
        values_by_agent: dict[str, list[float]] = {agent: [] for agent in self.env.possible_agents}
        dones_by_agent: dict[str, list[bool]] = {agent: [] for agent in self.env.possible_agents}
        episode_rows: list[dict[str, float | int]] = []

        for episode_offset in range(self.episodes_per_update):
            episode_id = start_episode + episode_offset
            observations, _ = self.env.reset(seed=self.seed + episode_id)
            guidance_decision: GuidanceDecision | None = None
            if self.guidance.should_update(episode_id):
                guidance_decision = self.guidance.get_guidance(
                    episode=episode_id,
                    state_snapshot=self.env.state_snapshot(),
                )
            done = False
            step_count = 0
            episode_return = 0.0
            collision_sum = 0.0
            final_coverage = float("nan")

            while not done and self.env.env.agents:
                action_dict: dict[str, np.ndarray] = {}
                log_prob_dict: dict[str, float] = {}
                value_dict: dict[str, float] = {}

                for agent in self.env.env.agents:
                    obs = np.asarray(observations[agent], dtype=np.float32)
                    obs_tensor = tensor(obs[None, :], self.device)
                    with torch.no_grad():
                        action_tensor, log_prob_tensor = self.actor.act(obs_tensor)
                        value_tensor = self.critic(obs_tensor)

                    action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
                    action_dict[agent] = action
                    log_prob_dict[agent] = float(log_prob_tensor.item())
                    value_dict[agent] = float(value_tensor.item())

                    obs_by_agent[agent].append(obs)
                    actions_by_agent[agent].append(action)
                    log_probs_by_agent[agent].append(log_prob_dict[agent])
                    values_by_agent[agent].append(value_dict[agent])

                observations, rewards, terminations, truncations, _ = self.env.step(action_dict)
                metrics = self.env.cooperation_metrics()
                shaping_rewards = self.guidance.shaping_rewards(
                    state_snapshot=self.env.state_snapshot(),
                    decision=guidance_decision,
                )
                done = all(terminations.values()) or all(truncations.values())

                for agent in action_dict:
                    shaped_reward = float(rewards.get(agent, 0.0)) + float(
                        shaping_rewards.get(agent, 0.0)
                    )
                    rewards_by_agent[agent].append(shaped_reward)
                    dones_by_agent[agent].append(done)

                episode_return += float(sum(rewards.values()))
                collision_sum += float(metrics["collision_count"])
                final_coverage = float(metrics["coverage_distance"])
                step_count += 1

            episode_rows.append(
                {
                    "episode": episode_id,
                    "seed": self.seed,
                    "steps": step_count,
                    "episode_return": episode_return,
                    "coverage_distance": final_coverage,
                    "collision_rate": collision_sum / max(step_count, 1),
                    "method": self.method,
                }
            )

        flat_obs: list[np.ndarray] = []
        flat_critic_obs: list[np.ndarray] = []
        flat_actions: list[np.ndarray] = []
        flat_log_probs: list[float] = []
        flat_returns: list[float] = []
        flat_advantages: list[float] = []

        for agent in self.env.possible_agents:
            if not rewards_by_agent[agent]:
                continue
            advantages, returns = compute_gae(
                rewards=rewards_by_agent[agent],
                values=values_by_agent[agent],
                dones=dones_by_agent[agent],
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )
            flat_obs.extend(obs_by_agent[agent])
            flat_critic_obs.extend(obs_by_agent[agent])
            flat_actions.extend(actions_by_agent[agent])
            flat_log_probs.extend(log_probs_by_agent[agent])
            flat_returns.extend(returns.tolist())
            flat_advantages.extend(advantages.tolist())

        return (
            RolloutBatch(
                obs=np.asarray(flat_obs, dtype=np.float32),
                critic_obs=np.asarray(flat_critic_obs, dtype=np.float32),
                actions=np.asarray(flat_actions, dtype=np.float32),
                log_probs=np.asarray(flat_log_probs, dtype=np.float32),
                returns=np.asarray(flat_returns, dtype=np.float32),
                advantages=np.asarray(flat_advantages, dtype=np.float32),
            ),
            episode_rows,
        )

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        """Run PPO updates over a flattened rollout batch."""
        obs = tensor(batch.obs, self.device)
        actions = tensor(batch.actions, self.device)
        old_log_probs = tensor(batch.log_probs, self.device)
        returns = tensor(batch.returns, self.device)
        advantages = normalize(tensor(batch.advantages, self.device))

        batch_size = obs.shape[0]
        indices = np.arange(batch_size)
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropy_losses: list[float] = []

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, self.minibatch_size):
                mb_idx = indices[start : start + self.minibatch_size]
                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                new_log_probs, entropy = self.actor.log_prob_entropy(mb_obs, mb_actions)
                values = self.critic(mb_obs)

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                unclipped = ratio * mb_advantages
                clipped = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * mb_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(values, mb_returns)
                entropy_loss = entropy.mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropy_losses.append(float(entropy_loss.item()))

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropy_losses)),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        """Save actor, critic, optimizer, and config."""
        output_path = ensure_parent_dir(path)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config,
                "method": self.method,
            },
            output_path,
        )

    def train(self) -> list[dict[str, float | int]]:
        """Train IPPO and return episode-level logs."""
        all_rows: list[dict[str, float | int]] = []
        best_return = -float("inf")

        for start_episode in range(0, self.total_episodes, self.episodes_per_update):
            batch, episode_rows = self.collect_rollouts(start_episode)
            losses = self.update(batch)
            recent_return = float(np.mean([row["episode_return"] for row in episode_rows]))

            for row in episode_rows:
                row.update(losses)
            append_dict_rows(self.log_path, episode_rows)
            all_rows.extend(episode_rows)

            if recent_return > best_return:
                best_return = recent_return
                self.save_checkpoint(self.checkpoint_path)

            if start_episode % max(self.log_interval, 1) == 0:
                print(
                    f"[IPPO] episode={start_episode + len(episode_rows):04d} "
                    f"return={recent_return:.3f} "
                    f"coverage={np.mean([row['coverage_distance'] for row in episode_rows]):.3f}"
                )

        return all_rows

    def close(self) -> None:
        self.env.close()
