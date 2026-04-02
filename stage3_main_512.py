import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

import pandas as pd

from stage3_onh3d_contract import (
    BMOPlane,
    ILMSurfaceModel,
    ONH3DCase,
    ONH3DQCMeta,
    SectorReference,
    TransformInfo,
    validate_onh3d_case,
)
from stage3_export_512 import (
    build_master_table_512,
    build_run_summary_df_512,
    export_results_excel_512,
)
from stage3_onh3d_metrics import ONH3DMetricConfig, compute_onh3d_metrics
from stage3_onh3d_report_adapter import adapt_onh3d_metrics_to_stage3_tables


def parse_args_512(argv=None):
    parser = argparse.ArgumentParser(description="Stage3 512 ONH3D pipeline entry")
    parser.add_argument("--onh3d-case", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _as_vector3(value: Any, default: Optional[np.ndarray] = None) -> np.ndarray:
    if value is None:
        if default is None:
            raise ValueError("Expected a 3D vector value")
        return np.asarray(default, dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError("Expected a finite vector with shape (3,)")
    return arr


def _normalize(vec: Any, default: Optional[np.ndarray] = None) -> np.ndarray:
    arr = _as_vector3(vec, default=default)
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        if default is not None:
            arr = np.asarray(default, dtype=float)
            norm = float(np.linalg.norm(arr))
        if norm <= 0:
            raise ValueError("Zero-length vector is not allowed")
    return arr / norm


def _derive_bmo_plane(bmo_ring_3d: np.ndarray, payload: Dict[str, Any], notes: list[str]) -> BMOPlane:
    plane_payload = payload.get("bmo_plane") or {}
    origin_default = np.mean(bmo_ring_3d, axis=0)
    p0 = bmo_ring_3d[0]
    p1 = bmo_ring_3d[1]
    p2 = bmo_ring_3d[2]
    normal_guess = np.cross(p1 - p0, p2 - p0)
    if float(np.linalg.norm(normal_guess)) <= 0:
        normal_guess = np.array([0.0, 0.0, 1.0], dtype=float)
    normal_default = _normalize(normal_guess, default=np.array([0.0, 0.0, 1.0], dtype=float))
    x_guess = p0 - origin_default
    if float(np.linalg.norm(x_guess)) <= 0:
        x_guess = np.array([1.0, 0.0, 0.0], dtype=float)
    x_guess = x_guess - np.dot(x_guess, normal_default) * normal_default
    if float(np.linalg.norm(x_guess)) <= 0:
        x_guess = np.array([1.0, 0.0, 0.0], dtype=float)
    x_default = _normalize(x_guess, default=np.array([1.0, 0.0, 0.0], dtype=float))
    y_default = _normalize(np.cross(normal_default, x_default), default=np.array([0.0, 1.0, 0.0], dtype=float))
    x_default = _normalize(np.cross(y_default, normal_default), default=x_default)

    if "bmo_plane" not in payload:
        notes.append("bmo_plane_missing_used_ring_derived_reference_frame")

    return BMOPlane(
        origin_3d=_as_vector3(plane_payload.get("origin_3d"), default=origin_default),
        normal_3d=_normalize(plane_payload.get("normal_3d"), default=normal_default),
        x_axis_3d=_normalize(plane_payload.get("x_axis_3d"), default=x_default),
        y_axis_3d=_normalize(plane_payload.get("y_axis_3d"), default=y_default),
    )


def _build_sector_reference(payload: Dict[str, Any], bmo_plane: BMOPlane, notes: list[str]) -> SectorReference:
    sector_payload = payload.get("sector_reference") or {}
    if "sector_reference" not in payload:
        notes.append("sector_reference_missing_used_bmo_plane_x_axis")
    kwargs = {
        "angle_zero_axis_3d": _normalize(
            sector_payload.get("angle_zero_axis_3d"),
            default=np.asarray(bmo_plane.x_axis_3d, dtype=float),
        ),
        "clockwise_positive": bool(sector_payload.get("clockwise_positive", True)),
    }
    if sector_payload.get("sector_8_bounds") is not None:
        kwargs["sector_8_bounds"] = sector_payload.get("sector_8_bounds")
    if sector_payload.get("sector_4_map") is not None:
        kwargs["sector_4_map"] = sector_payload.get("sector_4_map")
    if sector_payload.get("sector_2_map") is not None:
        kwargs["sector_2_map"] = sector_payload.get("sector_2_map")
    return SectorReference(
        **kwargs,
    )


def _build_transform_info(payload: Dict[str, Any], notes: list[str]) -> TransformInfo:
    transform_payload = payload.get("transform_info") or {}
    if "transform_info" not in payload:
        notes.append("transform_info_missing_defaulted_to_mm_world")
    return TransformInfo(
        world_unit=str(transform_payload.get("world_unit", "mm")),
        voxel_spacing_mm_xyz=tuple(transform_payload["voxel_spacing_mm_xyz"])
        if transform_payload.get("voxel_spacing_mm_xyz") is not None
        else None,
        voxel_to_world=transform_payload.get("voxel_to_world"),
    )


def _build_ilm_surface(payload: Dict[str, Any], notes: list[str]) -> ILMSurfaceModel:
    ilm_payload = payload.get("ilm_surface")
    if ilm_payload is None:
        raise ValueError("ilm_surface is required")
    if "vertex_normals_3d" not in ilm_payload:
        notes.append("ilm_surface.vertex_normals_3d_missing_left_as_none")
    if "surface_bounds_3d" not in ilm_payload:
        notes.append("ilm_surface.surface_bounds_3d_missing_left_as_none")
    return ILMSurfaceModel(
        vertices_3d=ilm_payload.get("vertices_3d"),
        faces=ilm_payload.get("faces"),
        vertex_normals_3d=ilm_payload.get("vertex_normals_3d"),
        surface_bounds_3d=ilm_payload.get("surface_bounds_3d"),
    )


def _build_qc_meta(
    payload: Dict[str, Any],
    bmo_ring_3d: np.ndarray,
    ilm_surface: ILMSurfaceModel,
    notes: list[str],
) -> ONH3DQCMeta:
    qc_payload = payload.get("qc_meta") or {}
    if "qc_meta" not in payload:
        notes.append("qc_meta_missing_populated_with_loader_defaults")

    bmo_ring_closed = bool(qc_payload.get("bmo_ring_closed", True))
    if not bmo_ring_closed:
        notes.append("qc_meta_reports_bmo_ring_closed_false_loader_kept_input_ring_as_ordered_closed_contract")

    ilm_vertices = np.asarray(ilm_surface.vertices_3d, dtype=float)
    ilm_faces = np.asarray(ilm_surface.faces)
    merged_notes = list(qc_payload.get("notes", [])) + list(notes)
    return ONH3DQCMeta(
        bmo_ring_closed=bmo_ring_closed,
        bmo_ring_sampling_count=int(qc_payload.get("bmo_ring_sampling_count", bmo_ring_3d.shape[0])),
        ilm_surface_vertex_count=int(qc_payload.get("ilm_surface_vertex_count", ilm_vertices.shape[0])),
        ilm_surface_face_count=int(qc_payload.get("ilm_surface_face_count", ilm_faces.shape[0])),
        bmo_plane_fit_rms_mm=qc_payload.get("bmo_plane_fit_rms_mm"),
        notes=merged_notes,
    )


def load_onh3d_case(case_path: Optional[str]) -> ONH3DCase:
    """Load a normalized 512 ONH3D JSON case into the canonical contract.

    Preferred input is the canonical Stage1/2 `onh3d_case.json` artifact with
    explicit `bmo_plane`, `sector_reference`, `transform_info`, and `qc_meta`.
    The loader still supports minimal omissions as a compatibility fallback,
    but those derived defaults are a secondary path and are recorded in
    `source_meta["loader_defaults_applied"]`.
    """

    if not case_path:
        raise ValueError("--onh3d-case is required")

    path = Path(case_path)
    if not path.is_file():
        raise FileNotFoundError(f"ONH3D case file not found: {case_path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    bmo_ring_3d = np.asarray(payload.get("bmo_ring_3d"), dtype=float)
    if bmo_ring_3d.ndim != 2 or bmo_ring_3d.shape[1] != 3 or bmo_ring_3d.shape[0] < 3:
        raise ValueError("bmo_ring_3d is required and must have shape (N,3) with N>=3")

    notes: list[str] = []
    bmo_plane = _derive_bmo_plane(bmo_ring_3d, payload, notes)
    sector_reference = _build_sector_reference(payload, bmo_plane, notes)
    transform_info = _build_transform_info(payload, notes)
    ilm_surface = _build_ilm_surface(payload, notes)
    qc_meta = _build_qc_meta(payload, bmo_ring_3d, ilm_surface, notes)

    source_meta = dict(payload.get("source_meta") or {})
    source_meta["loader_contract"] = "onh3d_case_json_v1"
    source_meta["loader_path"] = str(path)
    if notes:
        source_meta["loader_defaults_applied"] = list(notes)

    case = ONH3DCase(
        case_id=str(payload.get("case_id", "")).strip(),
        patient_id=str(payload.get("patient_id", "")).strip(),
        laterality=str(payload.get("laterality", "")).strip(),
        axial_length=float(payload.get("axial_length")),
        algorithm_version=str(payload.get("algorithm_version", "")).strip(),
        geometry_model=str(payload.get("geometry_model", "true_3d_onh")).strip(),
        bmo_ring_3d=bmo_ring_3d,
        bmo_plane=bmo_plane,
        sector_reference=sector_reference,
        transform_info=transform_info,
        ilm_surface=ilm_surface,
        qc_meta=qc_meta,
        diagnosis=payload.get("diagnosis"),
        stage=payload.get("stage"),
        source_meta=source_meta,
    )
    validate_onh3d_case(case)
    return case


def run_onh3d_stage3(
    case: ONH3DCase,
    *,
    config: Optional[ONH3DMetricConfig] = None,
    baseline_row: Optional[Dict[str, Any]] = None,
    self_check_df: Optional[pd.DataFrame] = None,
    workbook_path: Optional[str] = None,
    qc_slice_dir: Optional[str] = None,
    verbose: bool = False,
):
    metrics_result = compute_onh3d_metrics(case, config=config)
    adapted = adapt_onh3d_metrics_to_stage3_tables(case, metrics_result)
    del baseline_row, self_check_df

    master_df = build_master_table_512(
        case,
        metrics_result,
        adapted["mrw_df"],
        adapted["mra_df"],
        adapted["sector_df"],
        adapted["report_meta"],
    )
    run_summary_df = build_run_summary_df_512(
        case,
        metrics_result,
        adapted["report_meta"],
        output_path=workbook_path or "",
    )
    if workbook_path:
        export_results_excel_512(
            workbook_path,
            master_df,
            run_summary_df,
            adapted["mrw_df"],
            adapted["mra_df"],
            adapted["sector_df"],
        )
        if verbose:
            print(f"[stage3_512] wrote_excel={workbook_path}")
    if verbose:
        print(
            "[stage3_512] "
            f"case_id={case.case_id} valid_samples={metrics_result.valid_sample_count}/"
            f"{metrics_result.total_sample_count} "
            f"MRW_global_mean_um={metrics_result.MRW_global_mean_um} "
            f"MRA_global_sum_mm2={metrics_result.MRA_global_sum_mm2}"
        )
    return {
        "master_df": master_df,
        "run_summary_df": run_summary_df,
        "report_meta": adapted["report_meta"],
        "metrics_result": metrics_result,
    }


def main_512(argv=None):
    args = parse_args_512(argv)
    case = load_onh3d_case(args.onh3d_case)
    if args.case_id:
        case.case_id = str(args.case_id)
    validate_onh3d_case(case)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = output_dir / f"{case.case_id}_stage3_512.xlsx"
    if workbook_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output workbook already exists: {workbook_path}. Use --overwrite to replace it."
        )
    return run_onh3d_stage3(
        case,
        workbook_path=str(workbook_path),
        qc_slice_dir=args.output_dir,
        verbose=bool(args.verbose),
    )


if __name__ == "__main__":
    main_512()
