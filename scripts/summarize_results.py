"""Summarize experiment artifacts into a compact markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment outputs.")
    parser.add_argument(
        "--log_glob",
        default="logs/*_seed_*.csv",
        help="Glob used to discover training logs.",
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path("results/runtime_check.json"),
        help="Runtime verification JSON to include when present.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/experiment_summary.md"),
        help="Markdown summary path.",
    )
    parser.add_argument(
        "--latest_run_only",
        action="store_true",
        help="Use only the latest contiguous run from each log file.",
    )
    return parser.parse_args()


def latest_run_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only the latest run from a log with appended reruns."""
    if frame.empty or "episode" not in frame.columns:
        return frame
    reset_points = frame["episode"].diff().fillna(0) < 0
    if not reset_points.any():
        return frame
    start_idx = reset_points[reset_points].index[-1]
    return frame.loc[start_idx:].reset_index(drop=True)


def summarize_logs(log_paths: list[Path], latest_run_only: bool = False) -> str:
    if not log_paths:
        return "No training logs found."

    frames = []
    for path in log_paths:
        frame = pd.read_csv(path)
        frames.append(latest_run_frame(frame) if latest_run_only else frame)
    df = pd.concat(frames, ignore_index=True)
    summary = (
        df.groupby("method")
        .agg(
            runs=("seed", "nunique"),
            episodes=("episode", "count"),
            mean_return=("episode_return", "mean"),
            mean_coverage=("coverage_distance", "mean"),
            mean_collision=("collision_rate", "mean"),
        )
        .reset_index()
    )
    lines = ["| method | runs | episodes | mean_return | mean_coverage | mean_collision |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples(index=False):
        lines.append(
            "| "
            f"{row.method} | {row.runs} | {row.episodes} | "
            f"{row.mean_return:.3f} | {row.mean_coverage:.3f} | {row.mean_collision:.3f} |"
        )
    return "\n".join(lines)


def summarize_runtime(path: Path) -> str:
    if not path.exists():
        return "Runtime verification file not found."
    payload = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        f"- platform: `{payload.get('platform', 'unknown')}`",
        f"- python: `{payload.get('python', 'unknown')}`",
        f"- torch: `{payload.get('torch_version', 'unknown')}`",
        f"- cuda_available: `{payload.get('cuda_available', False)}`",
    ]
    if payload.get("cuda_device"):
        lines.append(f"- cuda_device: `{payload['cuda_device']}`")
    config_devices = payload.get("config_devices", {})
    for config_path, summary in sorted(config_devices.items()):
        lines.append(
            f"- {config_path}: `{summary.get('device', 'unknown')}` / "
            f"`{summary.get('device_name', 'unknown')}`"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    log_paths = sorted(Path().glob(args.log_glob))
    checkpoint_paths = sorted(Path("checkpoints").glob("*_best.pt"))
    evaluation_paths = sorted(Path("results").glob("*eval*.csv"))
    figure_paths = sorted(Path("results").glob("*.png"))

    output = "\n".join(
        [
            "# Experiment Summary",
            "",
            "## Runtime",
            summarize_runtime(args.runtime),
            "",
            "## Training Logs",
            summarize_logs(log_paths, latest_run_only=args.latest_run_only),
            "",
            "## Artifact Counts",
            f"- logs: `{len(log_paths)}`",
            f"- checkpoints: `{len(checkpoint_paths)}`",
            f"- evaluations: `{len(evaluation_paths)}`",
            f"- figures: `{len(figure_paths)}`",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n", encoding="utf-8")
    print(f"Saved summary to {args.output}")


if __name__ == "__main__":
    main()
