#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
pip install -r requirements.txt

python - <<'PY'
import platform
import sys

import torch

print(f"python={sys.version.split()[0]}")
print(f"platform={platform.platform()}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
else:
    print("cuda_device=no-gpu")
PY
