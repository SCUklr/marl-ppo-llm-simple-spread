"""Generate result plots from training logs."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Simple Spread training results.")
    parser.add_argument(
        "--log_paths",
        nargs="+",
        default=[
            "logs/*_seed_*.csv",
        ],
        help="Training log CSV files or glob patterns.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    return parser.parse_args()


def load_logs(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        matches = sorted(glob.glob(path))
        if not matches and Path(path).exists():
            matches = [path]
        for match in matches:
            frames.append(pd.read_csv(match))
    if not frames:
        raise FileNotFoundError("No log files found. Run training before plotting.")
    return pd.concat(frames, ignore_index=True)


def save_metric_plot(df: pd.DataFrame, metric: str, output_path: Path, ylabel: str) -> None:
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="episode", y=metric, hue="method", errorbar="sd")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_summary_table(df: pd.DataFrame, output_path: Path) -> None:
    summary = (
        df.groupby("method")
        .agg(
            final_return=("episode_return", "mean"),
            coverage_distance=("coverage_distance", "mean"),
            collision_rate=("collision_rate", "mean"),
        )
        .reset_index()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    df = load_logs(args.log_paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_metric_plot(
        df,
        metric="episode_return",
        output_path=args.output_dir / "learning_curves.png",
        ylabel="Episode Return",
    )
    save_metric_plot(
        df,
        metric="coverage_distance",
        output_path=args.output_dir / "coverage_distance.png",
        ylabel="Coverage Distance",
    )
    save_metric_plot(
        df,
        metric="collision_rate",
        output_path=args.output_dir / "collision_rate.png",
        ylabel="Collision Rate",
    )
    save_summary_table(df, args.output_dir / "comparison_table.csv")
    print(f"Saved plots and summary to {args.output_dir}")


if __name__ == "__main__":
    main()
