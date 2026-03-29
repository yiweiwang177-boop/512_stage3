#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy pandas scipy matplotlib openpyxl xlrd opencv-python-headless

python - <<'PY'
import importlib
import platform
import sys

print(f"python={sys.version.split()[0]}")
print(f"platform={platform.platform()}")
for name in ["numpy", "pandas", "scipy", "matplotlib", "openpyxl", "cv2"]:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "unknown")
    print(f"{name}={version}")
try:
    import torch
    print(f"torch={torch.__version__}")
except Exception:
    print("torch=not_installed")
PY

echo "Bootstrap complete: $REPO_ROOT/$VENV_DIR"
