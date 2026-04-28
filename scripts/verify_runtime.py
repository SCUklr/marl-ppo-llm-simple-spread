"""Verify Python, PyTorch, and configured training devices."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.algorithms.common import device_from_config, device_summary
from src.utils import ensure_parent_dir, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the experiment runtime.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/ippo.yaml",
            "configs/mappo.yaml",
            "configs/llm_guidance.yaml",
        ],
        help="Training configs to inspect.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/runtime_check.json"),
        help="Where to save the runtime summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, object] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "config_devices": {},
    }
    if torch.cuda.is_available():
        report["cuda_device"] = torch.cuda.get_device_name(0)
        report["cuda_count"] = torch.cuda.device_count()

    config_devices: dict[str, dict[str, object]] = {}
    for config_path in args.configs:
        config = load_yaml(config_path)
        device = device_from_config(config.get("training", {}))
        config_devices[config_path] = device_summary(device)

    report["config_devices"] = config_devices
    output_path = ensure_parent_dir(args.output)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"platform={report['platform']}")
    print(f"python={report['python']}")
    print(f"torch={report['torch_version']}")
    print(f"cuda_available={report['cuda_available']}")
    if "cuda_device" in report:
        print(f"cuda_device={report['cuda_device']}")
    for config_path, summary in config_devices.items():
        print(f"{config_path}: device={summary['device']} name={summary['device_name']}")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
