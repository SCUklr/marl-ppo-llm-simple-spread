"""Test the configured LLM guidance provider without running training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.guidance import GuidanceProvider
from src.utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test LLM guidance provider.")
    parser.add_argument("--config", type=Path, default=Path("configs/llm_guidance.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    provider = GuidanceProvider(config.get("llm_guidance", {}))
    state_snapshot = {
        "agents": {
            "agent_0": np.array([-0.8, 0.2], dtype=np.float32),
            "agent_1": np.array([0.1, -0.6], dtype=np.float32),
            "agent_2": np.array([0.7, 0.5], dtype=np.float32),
        },
        "landmarks": {
            "landmark_0": np.array([-0.9, 0.0], dtype=np.float32),
            "landmark_1": np.array([0.2, -0.8], dtype=np.float32),
            "landmark_2": np.array([0.9, 0.7], dtype=np.float32),
        },
    }
    decision = provider.get_guidance(episode=0, state_snapshot=state_snapshot)
    print(f"source={decision.source}")
    print(f"assignments={decision.assignments}")
    print(f"rationale={decision.rationale}")


if __name__ == "__main__":
    main()
