"""Optional high-level guidance for Simple Spread.

The module is intentionally safe to run without API keys. It supports a
deterministic heuristic fallback and OpenAI-compatible chat APIs such as Qwen.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


@dataclass(frozen=True)
class GuidanceDecision:
    """Agent-to-landmark assignment plus human-readable rationale."""

    assignments: dict[str, str]
    rationale: str
    source: str


class GuidanceProvider:
    """Provide high-level guidance at episode reset or fixed intervals."""

    def __init__(self, config: dict[str, Any]) -> None:
        load_dotenv()
        self.enabled = bool(config.get("enabled", False))
        self.provider = str(config.get("provider", "heuristic"))
        self.call_frequency = int(config.get("call_frequency_episodes", 100))
        self.reward_shaping_coef = float(config.get("reward_shaping_coef", 0.0))
        self.api_key_env = str(config.get("api_key_env", "LLM_API_KEY"))
        self.base_url = str(
            config.get(
                "base_url",
                os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            )
        )
        self.model = str(config.get("model", os.getenv("LLM_MODEL", "qwen-plus")))

    def should_update(self, episode: int) -> bool:
        """Return whether guidance should be refreshed this episode."""
        return self.enabled and episode % max(self.call_frequency, 1) == 0

    def get_guidance(
        self,
        episode: int,
        state_snapshot: dict[str, dict[str, np.ndarray]],
    ) -> GuidanceDecision:
        """Return an assignment decision.

        The current implementation uses nearest-neighbor matching as a fallback
        that does not require network access or paid API calls.
        """
        agents = state_snapshot.get("agents", {})
        landmarks = state_snapshot.get("landmarks", {})
        if not self.enabled or not agents or not landmarks:
            return GuidanceDecision({}, "Guidance disabled or state unavailable.", "none")

        if self.provider in {"qwen", "openai_compatible"}:
            decision = self._api_guidance(episode, agents, landmarks)
            if decision is not None:
                return decision

        assignments = self._nearest_unique_assignment(agents, landmarks)
        return GuidanceDecision(
            assignments=assignments,
            rationale=(
                "Fallback heuristic: assign each agent to a nearby distinct "
                "landmark to encourage coverage."
            ),
            source="heuristic",
        )

    def shaping_rewards(
        self,
        state_snapshot: dict[str, dict[str, np.ndarray]],
        decision: GuidanceDecision | None,
    ) -> dict[str, float]:
        """Convert assignment guidance into small distance-based shaping rewards."""
        if not self.enabled or self.reward_shaping_coef <= 0.0 or decision is None:
            return {}

        agents = state_snapshot.get("agents", {})
        landmarks = state_snapshot.get("landmarks", {})
        shaped: dict[str, float] = {}
        for agent, landmark in decision.assignments.items():
            if agent not in agents or landmark not in landmarks:
                continue
            distance = float(np.linalg.norm(agents[agent] - landmarks[landmark]))
            shaped[agent] = -self.reward_shaping_coef * distance
        return shaped

    @staticmethod
    def _nearest_unique_assignment(
        agents: dict[str, np.ndarray],
        landmarks: dict[str, np.ndarray],
    ) -> dict[str, str]:
        """Greedy nearest unique landmark assignment."""
        remaining = set(landmarks)
        assignments: dict[str, str] = {}
        for agent, agent_pos in agents.items():
            if not remaining:
                break
            best_landmark = min(
                remaining,
                key=lambda landmark: float(np.linalg.norm(agent_pos - landmarks[landmark])),
            )
            assignments[agent] = best_landmark
            remaining.remove(best_landmark)
        return assignments

    def _api_guidance(
        self,
        episode: int,
        agents: dict[str, np.ndarray],
        landmarks: dict[str, np.ndarray],
    ) -> GuidanceDecision | None:
        """Call an OpenAI-compatible chat API and parse assignment JSON."""
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            return None

        client = OpenAI(api_key=api_key, base_url=self.base_url)
        prompt = self._build_prompt(episode, agents, landmarks)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a high-level coordinator for a cooperative "
                            "multi-agent Simple Spread task. Return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            payload = json.loads(content)
        except Exception as exc:
            return GuidanceDecision(
                assignments=self._nearest_unique_assignment(agents, landmarks),
                rationale=f"API guidance failed; using heuristic fallback: {type(exc).__name__}",
                source="heuristic_fallback",
            )

        assignments = payload.get("assignments", {})
        valid_landmarks = set(landmarks)
        valid_agents = set(agents)
        clean_assignments = {
            str(agent): str(landmark)
            for agent, landmark in assignments.items()
            if str(agent) in valid_agents and str(landmark) in valid_landmarks
        }
        if not clean_assignments:
            clean_assignments = self._nearest_unique_assignment(agents, landmarks)

        return GuidanceDecision(
            assignments=clean_assignments,
            rationale=str(payload.get("rationale", "Qwen high-level assignment.")),
            source=self.provider,
        )

    @staticmethod
    def _build_prompt(
        episode: int,
        agents: dict[str, np.ndarray],
        landmarks: dict[str, np.ndarray],
    ) -> str:
        """Build a compact prompt that asks for structured sub-goals."""
        agent_lines = "\n".join(
            f"- {name}: [{pos[0]:.3f}, {pos[1]:.3f}]" for name, pos in agents.items()
        )
        landmark_lines = "\n".join(
            f"- {name}: [{pos[0]:.3f}, {pos[1]:.3f}]" for name, pos in landmarks.items()
        )
        return (
            f"Episode: {episode}\n"
            "Agents:\n"
            f"{agent_lines}\n"
            "Landmarks:\n"
            f"{landmark_lines}\n\n"
            "Assign each agent to a distinct landmark to minimize coverage distance "
            "and avoid duplicate targets. Return only JSON in this schema:\n"
            '{"assignments":{"agent_0":"landmark_0","agent_1":"landmark_1","agent_2":"landmark_2"},'
            '"rationale":"short explanation"}'
        )
