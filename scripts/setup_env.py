"""Create a virtual environment, install dependencies, and report runtime info."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path
from shutil import which


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap a project virtual environment.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to create the venv.")
    parser.add_argument("--venv", type=Path, default=Path(".venv"), help="Virtual environment directory.")
    parser.add_argument(
        "--backend",
        choices=["auto", "venv", "uv"],
        default="auto",
        help="Virtual environment backend. Use uv when system pip/venv is unavailable.",
    )
    return parser.parse_args()


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def select_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    return "uv" if which("uv") else "venv"


def main() -> None:
    args = parse_args()
    backend = select_backend(args.backend)
    if backend == "uv":
        run(["uv", "venv", "--python", args.python, str(args.venv)])
    else:
        run([args.python, "-m", "venv", str(args.venv)])

    if os.name == "nt":
        venv_python = args.venv / "Scripts" / "python.exe"
    else:
        venv_python = args.venv / "bin" / "python"

    if backend == "uv":
        run(
            [
                "uv",
                "pip",
                "install",
                "--link-mode=copy",
                "--python",
                str(venv_python),
                "-r",
                "requirements.txt",
            ]
        )
    else:
        run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"])
    run([str(venv_python), "scripts/verify_runtime.py"])

    print(f"platform={platform.platform()}")
    print(f"python={sys.version.split()[0]}")
    print(f"venv_python={venv_python}")
    print(f"backend={backend}")


if __name__ == "__main__":
    main()
