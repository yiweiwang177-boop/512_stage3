import re
import unicodedata
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd

from stage3_onh3d_contract import ONH3DCase
from stage3_onh3d_metrics import ONH3DMetricResult
from stage3_sector_reference import SECTOR_2_MAP, SECTOR_4_MAP, SECTOR_8_BOUNDS


def slugify_sector_label(label: Any) -> str:
    txt = unicodedata.normalize("NFKD", str(label)).encode("ascii", "ignore").decode("ascii")
    txt = txt.lower()
    txt = re.sub(r"[^a-z0-9]+", "_", txt)
    txt = re.sub(r"_+", "_", txt).strip("_")
    return txt or "unknown_sector"


def _ordered_unique_non_null(values):
    out = []
    seen = set()
    for value in values:
        if value is None:
            continue
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def get_expected_master_sector_schema_512() -> Dict[str, Any]:
    sector_8_labels = [name for _, _, name in SECTOR_8_BOUNDS]
    sector_4_labels = _ordered_unique_non_null(SECTOR_4_MAP.get(name) for name in sector_8_labels)
    sector_2_labels = _ordered_unique_non_null(SECTOR_2_MAP.get(name) for name in sector_8_labels)
    labels_by_level = {
        "sector_8": sector_8_labels,
        "sector_4": sector_4_labels,
        "sector_2": sector_2_labels,
    }

    ordered_columns = []
    column_map = {}
    specs = [
        ("MRW_um", "mean", "MRW", "um"),
        ("MRA_local_area_mm2", "sum", "MRA", "mm2"),
    ]
    for metric_name, aggregation_method, prefix, unit in specs:
        for level in ["sector_8", "sector_4", "sector_2"]:
            for sector_name in labels_by_level[level]:
                col = f"{prefix}_{level}_{slugify_sector_label(sector_name)}_{unit}"
                ordered_columns.append(col)
                column_map[(metric_name, aggregation_method, level, sector_name)] = col

    return {
        "labels_by_level": labels_by_level,
        "ordered_columns": ordered_columns,
        "column_map": column_map,
    }


def _build_master_sector_columns_512(sector_df: pd.DataFrame) -> Dict[str, float]:
    schema = get_expected_master_sector_schema_512()
    values = {col: np.nan for col in schema["ordered_columns"]}
    if sector_df is None or len(sector_df) == 0:
        return values

    for _, row in sector_df.iterrows():
        key = (
            row.get("metric_name"),
            row.get("aggregation_method"),
            row.get("level"),
            row.get("sector_name"),
        )
        col = schema["column_map"].get(key)
        if col is None:
            continue
        value = pd.to_numeric(pd.Series([row.get("value")]), errors="coerce").iloc[0]
        values[col] = float(value) if pd.notna(value) else np.nan
    return values


def build_master_table_512(
    case: ONH3DCase,
    metrics_result: ONH3DMetricResult,
    mrw_df: pd.DataFrame,
    mra_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    report_meta: Dict[str, Any],
) -> pd.DataFrame:
    del mrw_df, mra_df
    row = {
        "case_id": case.case_id,
        "patient_id": case.patient_id,
        "laterality": case.laterality,
        "axial_length": float(case.axial_length) if np.isfinite(case.axial_length) else np.nan,
        "algorithm_family": report_meta.get("algorithm_family"),
        "geometry_model": report_meta.get("geometry_model"),
        "algorithm_version": report_meta.get("algorithm_version"),
        "mra_sector_aggregation": report_meta.get("mra_sector_aggregation"),
        "valid_sample_count": int(metrics_result.valid_sample_count),
        "total_sample_count": int(metrics_result.total_sample_count),
        "ring_sample_count": int(metrics_result.config.ring_sample_count),
        "MRW_global_mean_um": metrics_result.MRW_global_mean_um,
        "MRW_global_min_um": metrics_result.MRW_global_min_um,
        "MRW_global_low10_mean_um": metrics_result.MRW_global_low10_mean_um,
        "MRA_global_sum_mm2": metrics_result.MRA_global_sum_mm2,
    }
    row.update(_build_master_sector_columns_512(sector_df))
    return pd.DataFrame([row])


def build_run_summary_df_512(
    case: ONH3DCase,
    metrics_result: ONH3DMetricResult,
    report_meta: Dict[str, Any],
    *,
    output_path: str = "",
    timestamp: str | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": timestamp or datetime.now().isoformat(timespec="seconds"),
                "case_id": case.case_id,
                "patient_id": case.patient_id,
                "laterality": case.laterality,
                "axial_length": case.axial_length,
                "algorithm_family": report_meta.get("algorithm_family"),
                "geometry_model": report_meta.get("geometry_model"),
                "algorithm_version": report_meta.get("algorithm_version"),
                "mra_sector_aggregation": report_meta.get("mra_sector_aggregation"),
                "valid_sample_count": int(metrics_result.valid_sample_count),
                "total_sample_count": int(metrics_result.total_sample_count),
                "MRW_global_mean_um": metrics_result.MRW_global_mean_um,
                "MRW_global_min_um": metrics_result.MRW_global_min_um,
                "MRW_global_low10_mean_um": metrics_result.MRW_global_low10_mean_um,
                "MRA_global_sum_mm2": metrics_result.MRA_global_sum_mm2,
                "excel_output": output_path,
            }
        ]
    )


def export_results_excel_512(
    output_path: str,
    master_df: pd.DataFrame,
    run_summary_df: pd.DataFrame,
    mrw_df: pd.DataFrame,
    mra_df: pd.DataFrame,
    sector_df: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        master_df.to_excel(writer, sheet_name="Master_Table", index=False)
        run_summary_df.to_excel(writer, sheet_name="Run_Summary", index=False)
        mrw_df.to_excel(writer, sheet_name="MRW_detail", index=False)
        mra_df.to_excel(writer, sheet_name="MRA_detail", index=False)
        sector_df.to_excel(writer, sheet_name="Sector_Summary", index=False)
