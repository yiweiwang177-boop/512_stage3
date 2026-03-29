#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${VENV_DIR:-.venv}"
if [ ! -d "$VENV_DIR" ]; then
  echo "Missing virtualenv: $VENV_DIR. Run scripts/bootstrap_cloud.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

STAGE2_JSON=""
BASE_TABLE=""
CASE_ID=""
PATIENT_ID=""
LATERALITY=""
OUTPUT_DIR=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage2-json) STAGE2_JSON="$2"; shift 2 ;;
    --base-table) BASE_TABLE="$2"; shift 2 ;;
    --case-id) CASE_ID="$2"; shift 2 ;;
    --patient-id) PATIENT_ID="$2"; shift 2 ;;
    --laterality) LATERALITY="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$STAGE2_JSON" ]]; then
  echo "Usage: scripts/run_stage3_cloud.sh --stage2-json <path> [--base-table <path>] [--output-dir <dir>] [--case-id <id>] [--patient-id <id>] [--laterality <L|R>] [extra args]" >&2
  exit 1
fi

CMD=(
  python zuizhong.py
  --input-mode stage2
  --stage2-json "$STAGE2_JSON"
)

if [[ -n "$CASE_ID" ]]; then
  CMD+=(--case-id "$CASE_ID")
fi

if [[ -n "$PATIENT_ID" ]]; then
  CMD+=(--patient-id "$PATIENT_ID")
fi

if [[ -n "$LATERALITY" ]]; then
  CMD+=(--laterality "$LATERALITY")
fi

if [[ -n "$BASE_TABLE" ]]; then
  CMD+=(--base-table "$BASE_TABLE")
fi

if [[ -n "$OUTPUT_DIR" ]]; then
  CMD+=(--output-dir "$OUTPUT_DIR")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Running Stage3 from: $REPO_ROOT"
echo "OCT_STAGE3_INPUT_ROOT=${OCT_STAGE3_INPUT_ROOT:-}"
echo "OCT_BASELINE_ROOT=${OCT_BASELINE_ROOT:-}"
echo "OCT_OUTPUT_ROOT=${OCT_OUTPUT_ROOT:-}"
python - "$STAGE2_JSON" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    print(f"json.case_id={payload.get('case_id')}")
    print(f"json.patient_id={payload.get('patient_id')}")
    print(f"json.laterality={payload.get('laterality')}")
except Exception as exc:
    print(f"json.identity_read_failed={exc}")
PY
"${CMD[@]}"
