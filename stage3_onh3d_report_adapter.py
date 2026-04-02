from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from stage3_onh3d_contract import ONH3DCase
from stage3_onh3d_metrics import ONH3DMetricResult


ONH3D_REPORT_META_FIELDS = [
    "algorithm_family",
    "geometry_model",
    "algorithm_version",
    "mra_sector_aggregation",
]

ONH3D_MRW_DETAIL_COLUMNS = [
    "sample_index",
    "anatomical_angle_deg",
    "mrw_len_um",
    "delta_s_mm",
    "phi_deg",
    "connection_valid",
    "connection_reject_reason",
    "bmo_x_mm",
    "bmo_y_mm",
    "bmo_z_mm",
    "ilm_hit_x_mm",
    "ilm_hit_y_mm",
    "ilm_hit_z_mm",
    "sector_8_name",
    "sector_4_name",
    "sector_2_name",
    "slice_id",
    "side",
    "scan_angle_deg",
    "bmo_x_px",
    "bmo_y_px",
    "ilm_x_px",
    "ilm_y_px",
]

ONH3D_MRA_DETAIL_COLUMNS = [
    "sample_index",
    "anatomical_angle_deg",
    "mrw_len_um",
    "delta_s_mm",
    "phi_deg",
    "local_area_mm2",
    "connection_valid",
    "bmo_x_mm",
    "bmo_y_mm",
    "bmo_z_mm",
    "ilm_hit_x_mm",
    "ilm_hit_y_mm",
    "ilm_hit_z_mm",
    "sector_8_name",
    "sector_4_name",
    "sector_2_name",
    "slice_id",
    "side",
    "scan_angle_deg",
    "bmo_x_px",
    "bmo_y_px",
    "ilm_hit_x_px",
    "ilm_hit_y_px",
]

ONH3D_SECTOR_SUMMARY_COLUMNS = [
    "source_table",
    "level",
    "sector_name",
    "metric_name",
    "value",
    "aggregation_method",
    "count",
]


def build_empty_onh3d_mrw_detail_df() -> pd.DataFrame:
    return pd.DataFrame(columns=ONH3D_MRW_DETAIL_COLUMNS)


def build_empty_onh3d_mra_detail_df() -> pd.DataFrame:
    return pd.DataFrame(columns=ONH3D_MRA_DETAIL_COLUMNS)


def build_empty_onh3d_sector_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=ONH3D_SECTOR_SUMMARY_COLUMNS)


def _xyz_or_nan(point: Optional[Any]) -> tuple[float, float, float]:
    if point is None:
        return np.nan, np.nan, np.nan
    arr = np.asarray(point, dtype=float).reshape(-1)
    if arr.shape[0] != 3 or not np.all(np.isfinite(arr)):
        return np.nan, np.nan, np.nan
    return float(arr[0]), float(arr[1]), float(arr[2])


def _sector_summary_rows(metrics_result: ONH3DMetricResult) -> list[Dict[str, Any]]:
    rows = []
    by_level = {
        "sector_8": metrics_result.sector_summary_8,
        "sector_4": metrics_result.sector_summary_4,
        "sector_2": metrics_result.sector_summary_2,
    }
    for level, system_summary in by_level.items():
        for label, summary in system_summary.items():
            rows.append(
                {
                    "source_table": "MRW_detail",
                    "level": level,
                    "sector_name": label,
                    "metric_name": "MRW_um",
                    "value": float(summary.mrw_mean_um) if np.isfinite(summary.mrw_mean_um) else np.nan,
                    "aggregation_method": "mean",
                    "count": int(summary.valid_sample_count),
                }
            )
            rows.append(
                {
                    "source_table": "MRW_detail",
                    "level": level,
                    "sector_name": label,
                    "metric_name": "MRW_low10_um",
                    "value": float(summary.mrw_low10_mean_um) if np.isfinite(summary.mrw_low10_mean_um) else np.nan,
                    "aggregation_method": "low_fraction_mean",
                    "count": int(summary.valid_sample_count),
                }
            )
            rows.append(
                {
                    "source_table": "MRA_detail",
                    "level": level,
                    "sector_name": label,
                    "metric_name": "MRA_local_area_mm2",
                    "value": float(summary.mra_sum_mm2) if np.isfinite(summary.mra_sum_mm2) else np.nan,
                    "aggregation_method": "sum",
                    "count": int(summary.valid_sample_count),
                }
            )
    return rows


def build_onh3d_report_meta(
    case: ONH3DCase,
    metrics_result: Optional[ONH3DMetricResult] = None,
) -> Dict[str, Any]:
    meta = {
        "algorithm_family": "ONH3D_512",
        "geometry_model": str(case.geometry_model),
        "algorithm_version": str(case.algorithm_version),
        "mra_sector_aggregation": "sum",
    }
    if metrics_result is not None:
        meta["valid_sample_count"] = int(metrics_result.valid_sample_count)
        meta["total_sample_count"] = int(metrics_result.total_sample_count)
        meta["ring_sample_count"] = int(metrics_result.config.ring_sample_count)
    return meta


def adapt_onh3d_metrics_to_stage3_tables(
    case: ONH3DCase,
    metrics_result: Optional[ONH3DMetricResult] = None,
) -> Dict[str, Any]:
    """Build Stage3-style report tables for the future 512 ONH3D pipeline.

    `phi_deg` is defined as the angle between the valid BMO-to-ILM connection
    vector and the local BMO ring tangent.

    For the 512 pipeline, MRA sector aggregation is explicitly `sum`. The
    future Sector_Summary should be generated by this adapter instead of
    reusing the legacy `build_sector_summary_from_tables()` MRA mean logic.

    The compatibility columns `slice_id`, `side`, and pixel-space coordinates
    remain in the table schema only for output compatibility. If a 512 result
    has no natural source for these values, they should stay null rather than
    being fabricated.

    This adapter is a pure mapping layer. It must not recompute `phi_deg`,
    low-fraction MRW summaries, or sector aggregation values from the core
    metrics detail.
    """
    mrw_df = build_empty_onh3d_mrw_detail_df()
    mra_df = build_empty_onh3d_mra_detail_df()
    sector_df = build_empty_onh3d_sector_summary_df()
    report_meta = build_onh3d_report_meta(case)

    # Metrics core integration is intentionally deferred. The first structural
    # milestone is to lock the shared output schema and metadata contract.
    if metrics_result is None:
        return {
            "mrw_df": mrw_df,
            "mra_df": mra_df,
            "sector_df": sector_df,
            "report_meta": report_meta,
        }

    mrw_rows = []
    mra_rows = []
    for metric in metrics_result.detail:
        bmo_x, bmo_y, bmo_z = _xyz_or_nan(metric.bmo_point_3d)
        ilm_x, ilm_y, ilm_z = _xyz_or_nan(metric.ilm_point_3d)
        mrw_rows.append(
            {
                "sample_index": int(metric.sample_idx),
                "anatomical_angle_deg": float(metric.theta_ref_deg) if np.isfinite(metric.theta_ref_deg) else np.nan,
                "mrw_len_um": float(metric.mrw_um) if np.isfinite(metric.mrw_um) else np.nan,
                "delta_s_mm": float(metric.delta_s_mm) if np.isfinite(metric.delta_s_mm) else np.nan,
                "phi_deg": float(metric.phi_deg) if np.isfinite(metric.phi_deg) else np.nan,
                "connection_valid": bool(metric.is_valid),
                "connection_reject_reason": metric.invalid_reason,
                "bmo_x_mm": bmo_x,
                "bmo_y_mm": bmo_y,
                "bmo_z_mm": bmo_z,
                "ilm_hit_x_mm": ilm_x,
                "ilm_hit_y_mm": ilm_y,
                "ilm_hit_z_mm": ilm_z,
                "sector_8_name": metric.sector_8,
                "sector_4_name": metric.sector_4,
                "sector_2_name": metric.sector_2,
                "slice_id": np.nan,
                "side": np.nan,
                "scan_angle_deg": np.nan,
                "bmo_x_px": np.nan,
                "bmo_y_px": np.nan,
                "ilm_x_px": np.nan,
                "ilm_y_px": np.nan,
            }
        )
        mra_rows.append(
            {
                "sample_index": int(metric.sample_idx),
                "anatomical_angle_deg": float(metric.theta_ref_deg) if np.isfinite(metric.theta_ref_deg) else np.nan,
                "mrw_len_um": float(metric.mrw_um) if np.isfinite(metric.mrw_um) else np.nan,
                "delta_s_mm": float(metric.delta_s_mm) if np.isfinite(metric.delta_s_mm) else np.nan,
                "phi_deg": float(metric.phi_deg) if np.isfinite(metric.phi_deg) else np.nan,
                "local_area_mm2": float(metric.mra_contrib_mm2) if np.isfinite(metric.mra_contrib_mm2) else np.nan,
                "connection_valid": bool(metric.is_valid),
                "bmo_x_mm": bmo_x,
                "bmo_y_mm": bmo_y,
                "bmo_z_mm": bmo_z,
                "ilm_hit_x_mm": ilm_x,
                "ilm_hit_y_mm": ilm_y,
                "ilm_hit_z_mm": ilm_z,
                "sector_8_name": metric.sector_8,
                "sector_4_name": metric.sector_4,
                "sector_2_name": metric.sector_2,
                "slice_id": np.nan,
                "side": np.nan,
                "scan_angle_deg": np.nan,
                "bmo_x_px": np.nan,
                "bmo_y_px": np.nan,
                "ilm_hit_x_px": np.nan,
                "ilm_hit_y_px": np.nan,
            }
        )

    mrw_df = pd.DataFrame(mrw_rows, columns=ONH3D_MRW_DETAIL_COLUMNS)
    mra_df = pd.DataFrame(mra_rows, columns=ONH3D_MRA_DETAIL_COLUMNS)
    sector_df = pd.DataFrame(_sector_summary_rows(metrics_result), columns=ONH3D_SECTOR_SUMMARY_COLUMNS)
    report_meta = build_onh3d_report_meta(case, metrics_result)
    return {
        "mrw_df": mrw_df,
        "mra_df": mra_df,
        "sector_df": sector_df,
        "report_meta": report_meta,
    }
