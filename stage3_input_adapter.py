import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


SUPPORTED_STAGE2_SCHEMA_VERSION = "v1"
REQUIRED_CASE_FIELDS = ["stage2_schema_version", "case_id", "patient_id", "laterality", "slices"]
REQUIRED_SLICE_FIELDS = [
    "scan_index",
    "slice_stem",
    "image_width",
    "image_height",
    "full_ilm_px",
    "bmo_px",
    "cutoff_px",
    "rnfl_effective_lower_px",
]


def _coerce_point(point: Any, field_name: str) -> Tuple[float, float]:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        raise ValueError(f"{field_name} point must be a 2-item list/tuple")
    try:
        return float(point[0]), float(point[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} point must be numeric") from exc


def _coerce_polyline(
    points: Any,
    field_name: str,
    *,
    min_points: int,
    exact_points: Optional[int] = None,
) -> List[Tuple[float, float]]:
    if not isinstance(points, list):
        raise ValueError(f"{field_name} must be a list of points")
    if exact_points is not None and len(points) != exact_points:
        raise ValueError(f"{field_name} must contain exactly {exact_points} points")
    if len(points) < min_points:
        raise ValueError(f"{field_name} must contain at least {min_points} points")
    return [_coerce_point(pt, field_name) for pt in points]


def normalize_stage2_slice_record(slice_raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(slice_raw, dict):
        raise ValueError("slice record must be a JSON object")

    normalized = {}
    for key in REQUIRED_SLICE_FIELDS:
        if key not in slice_raw:
            raise ValueError(f"missing slice field: {key}")

    try:
        normalized["scan_index"] = int(slice_raw["scan_index"])
        normalized["slice_stem"] = str(slice_raw["slice_stem"])
        normalized["image_width"] = int(slice_raw["image_width"])
        normalized["image_height"] = int(slice_raw["image_height"])
    except (TypeError, ValueError) as exc:
        raise ValueError("scan_index/image_width/image_height must be numeric") from exc

    normalized["full_ilm_px"] = _coerce_polyline(
        slice_raw["full_ilm_px"], "full_ilm_px", min_points=2
    )
    normalized["bmo_px"] = _coerce_polyline(
        slice_raw["bmo_px"], "bmo_px", min_points=2, exact_points=2
    )
    normalized["cutoff_px"] = _coerce_polyline(
        slice_raw["cutoff_px"], "cutoff_px", min_points=2, exact_points=2
    )
    normalized["rnfl_effective_lower_px"] = _coerce_polyline(
        slice_raw["rnfl_effective_lower_px"], "rnfl_effective_lower_px", min_points=2
    )
    normalized["review_status"] = slice_raw.get("review_status")
    normalized["source_flags"] = slice_raw.get("source_flags", {})
    normalized["image_path"] = slice_raw.get("image_path")
    normalized["source_path"] = slice_raw.get("source_path")
    return normalized


def load_stage2_case(stage2_json_path: str, expected_case_id: Optional[str] = None) -> Dict[str, Any]:
    if not stage2_json_path or not os.path.isfile(stage2_json_path):
        raise FileNotFoundError(f"Stage2 json not found: {stage2_json_path}")

    with open(stage2_json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Stage2 case JSON must be a JSON object")

    for key in REQUIRED_CASE_FIELDS:
        if key not in data:
            raise ValueError(f"missing case field: {key}")

    schema_version = str(data["stage2_schema_version"])
    if schema_version != SUPPORTED_STAGE2_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported stage2_schema_version: {schema_version}, "
            f"expected {SUPPORTED_STAGE2_SCHEMA_VERSION}"
        )

    case_id = str(data["case_id"])
    if expected_case_id is not None and str(expected_case_id) != case_id:
        raise ValueError(f"case_id mismatch: expected {expected_case_id}, got {case_id}")

    laterality = str(data["laterality"]).strip().upper()
    if laterality not in {"L", "R"}:
        raise ValueError(f"invalid laterality: {laterality}")

    raw_slices = data["slices"]
    if not isinstance(raw_slices, list):
        raise ValueError("slices must be a list")

    normalized_slices = [normalize_stage2_slice_record(item) for item in raw_slices]

    normalized_case = dict(data)
    normalized_case["stage2_schema_version"] = schema_version
    normalized_case["case_id"] = case_id
    normalized_case["patient_id"] = str(data["patient_id"])
    normalized_case["laterality"] = laterality
    normalized_case["slices"] = sorted(normalized_slices, key=lambda item: item["scan_index"])
    return normalized_case


def load_patient_baseline_row(
    base_table_path: str,
    patient_id: str,
    laterality: str,
) -> Dict[str, Any]:
    if not base_table_path or not os.path.isfile(base_table_path):
        raise FileNotFoundError(f"Base table not found: {base_table_path}")

    df = pd.read_excel(base_table_path)
    required_cols = ["Patient_ID", "Laterality", "Axial_Length"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"missing baseline columns: {missing_cols}")

    laterality = str(laterality).strip().upper()
    mask = (df["Patient_ID"].astype(str) == str(patient_id)) & (
        df["Laterality"].astype(str).str.strip().str.upper() == laterality
    )
    matched = df[mask]
    if matched.empty:
        raise ValueError(
            f"baseline row not found for Patient_ID={patient_id}, Laterality={laterality}"
        )

    row = matched.iloc[0].to_dict()
    row["Patient_ID"] = str(row.get("Patient_ID"))
    row["Laterality"] = laterality
    if pd.notna(row.get("Axial_Length")):
        row["Axial_Length"] = float(row["Axial_Length"])
    return row


def validate_stage3_input_contract(
    stage2_case: Dict[str, Any],
    baseline_row: Dict[str, Any],
) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []

    def add(check: str, status: str, detail: str) -> None:
        records.append({"Check": check, "Status": status, "Detail": detail})

    add("stage2:case_id", "PASS", str(stage2_case.get("case_id")))
    schema_version = stage2_case.get("stage2_schema_version")
    add(
        "stage2:schema_version_present",
        "PASS" if schema_version is not None else "FAIL",
        str(schema_version),
    )
    add(
        "stage2:schema_version_supported",
        "PASS" if schema_version == SUPPORTED_STAGE2_SCHEMA_VERSION else "FAIL",
        str(schema_version),
    )
    baseline_ok = bool(baseline_row) and str(baseline_row.get("Patient_ID", "")).strip() != ""
    add(
        "baseline:matched_patient_laterality",
        "PASS" if baseline_ok else "FAIL",
        f"Patient_ID={baseline_row.get('Patient_ID')}, Laterality={baseline_row.get('Laterality')}",
    )

    slices = stage2_case.get("slices", [])
    scan_indices = sorted(int(item["scan_index"]) for item in slices)
    expected = list(range(1, 13))
    add(
        "stage2:slice_count",
        "PASS" if len(slices) == 12 else "FAIL",
        f"count={len(slices)}",
    )
    add(
        "stage2:scan_indices_complete",
        "PASS" if scan_indices == expected else "FAIL",
        f"indices={scan_indices}",
    )

    required_fields_ok = True
    for item in slices:
        sid = int(item["scan_index"])
        for field in REQUIRED_SLICE_FIELDS:
            has_field = field in item and item[field] is not None
            add(
                f"slice:{sid}:{field}",
                "PASS" if has_field else "FAIL",
                "present" if has_field else "missing",
            )
            required_fields_ok = required_fields_ok and has_field

    widths = {int(item["image_width"]) for item in slices}
    heights = {int(item["image_height"]) for item in slices}
    add(
        "stage2:image_width_present",
        "PASS" if all(width > 0 for width in widths) else "FAIL",
        f"widths={sorted(widths)}",
    )
    add(
        "stage2:image_height_present",
        "PASS" if all(height > 0 for height in heights) else "FAIL",
        f"heights={sorted(heights)}",
    )

    add(
        "stage2:required_fields_complete",
        "PASS" if required_fields_ok else "FAIL",
        "validated required slice fields",
    )
    return records
