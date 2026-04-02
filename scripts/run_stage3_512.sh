#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${VENV_DIR:-.venv-stage3}"
if [ ! -d "$VENV_DIR" ]; then
  echo "Missing virtualenv: $VENV_DIR. Run scripts/setup_stage3_env.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

ONH3D_CASE=""
OUTPUT_DIR=""
CASE_ID=""
OVERWRITE=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --onh3d-case) ONH3D_CASE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --case-id) CASE_ID="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --verbose) VERBOSE=1; shift ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$ONH3D_CASE" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: scripts/run_stage3_512.sh --onh3d-case <path> --output-dir <dir> [--case-id <id>] [--overwrite] [--verbose]" >&2
  exit 1
fi

CMD=(python stage3_main_512.py --onh3d-case "$ONH3D_CASE" --output-dir "$OUTPUT_DIR")

if [[ -n "$CASE_ID" ]]; then
  CMD+=(--case-id "$CASE_ID")
fi
if [[ $OVERWRITE -eq 1 ]]; then
  CMD+=(--overwrite)
fi
if [[ $VERBOSE -eq 1 ]]; then
  CMD+=(--verbose)
fi

"${CMD[@]}"
