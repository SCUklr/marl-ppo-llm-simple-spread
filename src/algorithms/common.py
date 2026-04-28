"""Shared PPO components for IPPO and MAPPO-style training."""

from __future__ import annotations

from dataclasses import dataclass
import platform
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def mlp(input_dim: int, hidden_sizes: list[int], output_dim: int) -> nn.Sequential:
    """Create a small MLP with Tanh activations."""
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(last_dim, hidden_size))
        layers.append(nn.Tanh())
        last_dim = hidden_size
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """Gaussian actor for continuous PettingZoo MPE actions."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int],
        action_low: np.ndarray,
        action_high: np.ndarray,
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        self.net = mlp(obs_dim, hidden_sizes, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        action = torch.clamp(raw_action, self.action_low, self.action_high)
        return action, log_prob

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the clipped Gaussian mean for evaluation."""
        mean = self.net(obs)
        return torch.clamp(mean, self.action_low, self.action_high)

    def log_prob_entropy(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class ValueCritic(nn.Module):
    """Scalar value network."""

    def __init__(self, input_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        self.net = mlp(input_dim, hidden_sizes, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values).squeeze(-1)


@dataclass
class RolloutBatch:
    """Flattened PPO rollout data."""

    obs: np.ndarray
    critic_obs: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute generalized advantage estimates for one trajectory."""
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_gae = 0.0

    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            next_value = 0.0
            next_non_terminal = 1.0 - float(dones[step])
        else:
            next_value = values[step + 1]
            next_non_terminal = 1.0 - float(dones[step])
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[step] = last_gae

    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


def normalize(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize a tensor if it has more than one element."""
    if values.numel() <= 1:
        return values
    return (values - values.mean()) / (values.std() + eps)


def tensor(data: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert NumPy data to float tensor on target device."""
    return torch.as_tensor(data, dtype=torch.float32, device=device)


def device_from_config(config: dict[str, Any]) -> torch.device:
    """Pick CPU, CUDA, or MPS from config."""
    requested = str(config.get("device", "auto"))
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_summary(device: torch.device) -> dict[str, Any]:
    """Return a compact runtime summary for experiment logs."""
    summary: dict[str, Any] = {
        "device": str(device),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        summary["device_name"] = props.name
        summary["device_capability"] = f"{props.major}.{props.minor}"
    elif device.type == "mps":
        summary["device_name"] = "Apple Metal"
    else:
        summary["device_name"] = "CPU"
    return summary
