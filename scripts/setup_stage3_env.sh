#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv-stage3}"
REQ_FILE="${REQ_FILE:-requirements-stage3.txt}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -f "$REQ_FILE" ]; then
  echo "Missing requirements file: $REQ_FILE" >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$REQ_FILE"

python - <<'PY'
import importlib
for name in ["numpy", "pandas", "openpyxl"]:
    module = importlib.import_module(name)
    print(f"{name}={getattr(module, '__version__', 'unknown')}")
PY

echo "Stage3 512 environment is ready: $REPO_ROOT/$VENV_DIR"
