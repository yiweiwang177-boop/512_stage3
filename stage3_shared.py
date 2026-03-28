import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


NATIVE_SPACING_X_MM = 0.00586
NATIVE_SPACING_Z_MM = 0.00529
N_SLICES = 12


@dataclass
class CaseMeta:
    case_id: str
    patient_id: str
    laterality: str
    axial_length: float
    diagnosis: Optional[str]
    stage: Optional[str]
    image_width: int
    image_height: int
    x_center: float
    y_center: float
    n_slices: int = N_SLICES


@dataclass
class SliceMeta:
    slice_id: int
    slice_stem: str
    scan_index: int
    angle_deg: float
    image_shape: Tuple[int, int]
    x_center: float
    y_center: float
    full_ilm_px: List[Tuple[float, float]]
    bmo_left_px: Tuple[float, float]
    bmo_right_px: Tuple[float, float]
    cutoff_left_px: Tuple[float, float]
    cutoff_right_px: Tuple[float, float]
    rnfl_effective_lower_px: List[Tuple[float, float]]
    review_status: Optional[str]
    source_flags: Any
    image_path: Optional[str] = None
    source_path: Optional[str] = None


@dataclass
class Stage3SharedCase:
    case_meta: CaseMeta
    slice_meta_list: List[SliceMeta]
    compute_meta: Dict[str, Any]


def _build_scale_x(axial_length: float) -> float:
    return NATIVE_SPACING_X_MM * ((0.01306 * (axial_length - 1.82)) / (0.01306 * (24.0 - 1.82)))


def _image_center(image_width: int, image_height: int) -> Tuple[float, float]:
    return float(image_width) / 2.0, float(image_height) / 2.0


def _split_full_ilm_for_legacy(slice_meta: SliceMeta) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    pts = sorted(slice_meta.full_ilm_px, key=lambda item: item[0])
    left_cutoff_x = float(slice_meta.cutoff_left_px[0])
    right_cutoff_x = float(slice_meta.cutoff_right_px[0])

    left_pts = [pt for pt in pts if pt[0] <= left_cutoff_x]
    right_pts = [pt for pt in pts if pt[0] >= right_cutoff_x]

    if len(left_pts) < 2 or len(right_pts) < 2:
        x_center = float(slice_meta.image_shape[1]) / 2.0
        left_pts = [pt for pt in pts if pt[0] <= x_center]
        right_pts = [pt for pt in pts if pt[0] >= x_center]

    return left_pts, right_pts


def _pixel_to_3d(
    pixel_point: Tuple[float, float],
    *,
    x_center: float,
    scale_x: float,
    scale_z: float,
    angle_deg: float,
    laterality: str,
) -> Tuple[float, float, float]:
    pixel_x, pixel_y = pixel_point
    r_mm = (float(pixel_x) - float(x_center)) * float(scale_x)
    angle_rad = math.radians(float(angle_deg))
    x_3d = r_mm * math.cos(angle_rad)
    y_3d = r_mm * math.sin(angle_rad)
    z_3d = float(pixel_y) * float(scale_z)

    if str(laterality).upper() == "L":
        x_3d = -x_3d

    return float(x_3d), float(y_3d), float(z_3d)


def build_scale_x(axial_length: float) -> float:
    return _build_scale_x(axial_length)


def pixel_to_3d(
    pixel_point: Tuple[float, float],
    *,
    x_center: float,
    scale_x: float,
    scale_z: float,
    angle_deg: float,
    laterality: str,
) -> Tuple[float, float, float]:
    return _pixel_to_3d(
        pixel_point,
        x_center=x_center,
        scale_x=scale_x,
        scale_z=scale_z,
        angle_deg=angle_deg,
        laterality=laterality,
    )


def build_stage3_shared_structure(
    stage2_case: Dict[str, Any],
    baseline_row: Dict[str, Any],
) -> Stage3SharedCase:
    slices = sorted(stage2_case["slices"], key=lambda item: item["scan_index"])
    first = slices[0]

    case_meta = CaseMeta(
        case_id=str(stage2_case["case_id"]),
        patient_id=str(stage2_case["patient_id"]),
        laterality=str(stage2_case["laterality"]).strip().upper(),
        axial_length=float(baseline_row["Axial_Length"]),
        diagnosis=baseline_row.get("Diagnosis"),
        stage=baseline_row.get("Stage"),
        image_width=int(first["image_width"]),
        image_height=int(first["image_height"]),
        x_center=_image_center(int(first["image_width"]), int(first["image_height"]))[0],
        y_center=_image_center(int(first["image_width"]), int(first["image_height"]))[1],
        n_slices=N_SLICES,
    )

    slice_meta_list: List[SliceMeta] = []
    compute_meta = {
        "BMO_META": [],
        "ILM_FULL_META": [],
        "CUT_POINTS_META": [],
        "RNFL_EFFECTIVE_SEG_META": [],
        "SLICE_META": {},
    }

    for slice_raw in slices:
        bmo_sorted = sorted(slice_raw["bmo_px"], key=lambda item: item[0])
        cutoff_sorted = sorted(slice_raw["cutoff_px"], key=lambda item: item[0])
        scan_index = int(slice_raw["scan_index"])
        image_height = int(slice_raw["image_height"])
        image_width = int(slice_raw["image_width"])
        x_center, y_center = _image_center(image_width, image_height)

        slice_meta = SliceMeta(
            slice_id=scan_index,
            slice_stem=str(slice_raw["slice_stem"]),
            scan_index=scan_index,
            angle_deg=(scan_index - 1) * 15.0,
            image_shape=(image_height, image_width),
            x_center=x_center,
            y_center=y_center,
            full_ilm_px=list(slice_raw["full_ilm_px"]),
            bmo_left_px=tuple(bmo_sorted[0]),
            bmo_right_px=tuple(bmo_sorted[-1]),
            cutoff_left_px=tuple(cutoff_sorted[0]),
            cutoff_right_px=tuple(cutoff_sorted[-1]),
            rnfl_effective_lower_px=list(slice_raw["rnfl_effective_lower_px"]),
            review_status=slice_raw.get("review_status"),
            source_flags=slice_raw.get("source_flags", {}),
            image_path=slice_raw.get("image_path"),
            source_path=slice_raw.get("source_path"),
        )
        slice_meta_list.append(slice_meta)

        compute_meta["SLICE_META"][scan_index] = {
            "slice_id": slice_meta.slice_id,
            "slice_stem": slice_meta.slice_stem,
            "scan_index": slice_meta.scan_index,
            "angle_deg": slice_meta.angle_deg,
            "image_shape": slice_meta.image_shape,
            "image_width": slice_meta.image_shape[1],
            "image_height": slice_meta.image_shape[0],
            "x_center": slice_meta.x_center,
            "y_center": slice_meta.y_center,
            "rotation_center_px": [slice_meta.x_center, slice_meta.y_center],
            "center_source": "image_center",
            "review_status": slice_meta.review_status,
            "source_flags": slice_meta.source_flags,
            "image_path": slice_meta.image_path,
            "source_path": slice_meta.source_path,
        }
        compute_meta["BMO_META"].extend(
            [
                {
                    "slice_id": scan_index,
                    "scan_index": scan_index,
                    "side": "L",
                    "pixel_x": float(slice_meta.bmo_left_px[0]),
                    "pixel_y": float(slice_meta.bmo_left_px[1]),
                },
                {
                    "slice_id": scan_index,
                    "scan_index": scan_index,
                    "side": "R",
                    "pixel_x": float(slice_meta.bmo_right_px[0]),
                    "pixel_y": float(slice_meta.bmo_right_px[1]),
                },
            ]
        )
        compute_meta["CUT_POINTS_META"].extend(
            [
                {
                    "slice_id": scan_index,
                    "scan_index": scan_index,
                    "side": "L",
                    "pixel_x": float(slice_meta.cutoff_left_px[0]),
                    "pixel_y": float(slice_meta.cutoff_left_px[1]),
                },
                {
                    "slice_id": scan_index,
                    "scan_index": scan_index,
                    "side": "R",
                    "pixel_x": float(slice_meta.cutoff_right_px[0]),
                    "pixel_y": float(slice_meta.cutoff_right_px[1]),
                },
            ]
        )
        for px, py in slice_meta.full_ilm_px:
            compute_meta["ILM_FULL_META"].append(
                {
                    "slice_id": scan_index,
                    "scan_index": scan_index,
                    "pixel_x": float(px),
                    "pixel_y": float(py),
                }
            )
        for px, py in slice_meta.rnfl_effective_lower_px:
            compute_meta["RNFL_EFFECTIVE_SEG_META"].append(
                {
                    "slice_id": scan_index,
                    "scan_index": scan_index,
                    "pixel_x": float(px),
                    "pixel_y": float(py),
                }
            )

    return Stage3SharedCase(
        case_meta=case_meta,
        slice_meta_list=slice_meta_list,
        compute_meta=compute_meta,
    )


def build_legacy_cloud_from_shared(shared_case: Stage3SharedCase) -> Dict[str, Any]:
    """Build a temporary compatibility view from canonical shared data.

    This bridge exists for legacy/output compatibility only. New Stage3
    parameter code should treat `shared_case` and canonical `compute_meta`
    as the internal source of truth, and should not add fresh dependence on
    compat semantics such as ILM_ROI / ALI / ALCS.
    """
    case_meta = shared_case.case_meta
    scale_x = _build_scale_x(case_meta.axial_length)
    scale_z = NATIVE_SPACING_Z_MM

    cloud = {
        "case_id": case_meta.case_id,
        "patient_id": case_meta.patient_id,
        "input_mode": "stage2",
        "shared_case": shared_case,
        "BMO": [],
        "ALI": [],
        "ALCS": [],
        "ILM_ROI": [],
        "BMO_META": [],
        "ILM_META": [],
        "ALI_META": [],
        "ALCS_META": [],
        "SLICE_META": {},
        "laterality": case_meta.laterality,
        "axial_length": float(case_meta.axial_length),
        "z_stabilization_status": "inactive_stage2_no_z_correction",
        "rotation_center_source": "image_center",
        "rotation_center_case_px": [case_meta.x_center, case_meta.y_center],
    }

    for slice_meta in shared_case.slice_meta_list:
        cloud["SLICE_META"][slice_meta.slice_id] = {
            "slice_id": slice_meta.slice_id,
            "scan_index": slice_meta.scan_index,
            "image_path": slice_meta.image_path,
            "source_path": slice_meta.source_path,
            "slice_stem": slice_meta.slice_stem,
            "x_center": slice_meta.x_center,
            "y_center": slice_meta.y_center,
            "rotation_center_px": [slice_meta.x_center, slice_meta.y_center],
            "center_source": "image_center",
            "delta_z": 0.0,
            "scale_X": scale_x,
            "scale_Z": scale_z,
            "angle_deg": slice_meta.angle_deg,
            "laterality": case_meta.laterality,
            "image_shape": slice_meta.image_shape,
            "image_width": slice_meta.image_shape[1],
            "image_height": slice_meta.image_shape[0],
            "review_status": slice_meta.review_status,
            "source_flags": slice_meta.source_flags,
        }

        bmo_lr = [("L", slice_meta.bmo_left_px), ("R", slice_meta.bmo_right_px)]
        ali_lr = [("L", slice_meta.cutoff_left_px), ("R", slice_meta.cutoff_right_px)]

        for side, pt2d in bmo_lr:
            pt3d = _pixel_to_3d(
                pt2d,
                x_center=slice_meta.x_center,
                scale_x=scale_x,
                scale_z=scale_z,
                angle_deg=slice_meta.angle_deg,
                laterality=case_meta.laterality,
            )
            cloud["BMO"].append(pt3d)
            cloud["BMO_META"].append(
                {
                    "slice_id": slice_meta.slice_id,
                    "scan_index": slice_meta.scan_index,
                    "side": side,
                    "pixel_x": float(pt2d[0]),
                    "pixel_y": float(pt2d[1]),
                    "point_3d": pt3d,
                }
            )

        for side, pt2d in ali_lr:
            pt3d = _pixel_to_3d(
                pt2d,
                x_center=slice_meta.x_center,
                scale_x=scale_x,
                scale_z=scale_z,
                angle_deg=slice_meta.angle_deg,
                laterality=case_meta.laterality,
            )
            cloud["ALI"].append(pt3d)
            cloud["ALI_META"].append(
                {
                    "slice_id": slice_meta.slice_id,
                    "scan_index": slice_meta.scan_index,
                    "side": side,
                    "pixel_x": float(pt2d[0]),
                    "pixel_y": float(pt2d[1]),
                    "point_3d": pt3d,
                }
            )

        for pt2d in slice_meta.rnfl_effective_lower_px:
            pt3d = _pixel_to_3d(
                pt2d,
                x_center=slice_meta.x_center,
                scale_x=scale_x,
                scale_z=scale_z,
                angle_deg=slice_meta.angle_deg,
                laterality=case_meta.laterality,
            )
            cloud["ALCS"].append(pt3d)
            cloud["ALCS_META"].append(
                {
                    "slice_id": slice_meta.slice_id,
                    "scan_index": slice_meta.scan_index,
                    "pixel_x": float(pt2d[0]),
                    "pixel_y": float(pt2d[1]),
                    "point_3d": pt3d,
                }
            )

        left_ilm_pts, right_ilm_pts = _split_full_ilm_for_legacy(slice_meta)
        for pt2d in left_ilm_pts + right_ilm_pts:
            pt3d = _pixel_to_3d(
                pt2d,
                x_center=slice_meta.x_center,
                scale_x=scale_x,
                scale_z=scale_z,
                angle_deg=slice_meta.angle_deg,
                laterality=case_meta.laterality,
            )
            cloud["ILM_ROI"].append(pt3d)
            cloud["ILM_META"].append(
                {
                    "slice_id": slice_meta.slice_id,
                    "scan_index": slice_meta.scan_index,
                    "pixel_x": float(pt2d[0]),
                    "pixel_y": float(pt2d[1]),
                    "point_3d": pt3d,
                }
            )

    return cloud


def stage3_shared_case_to_dict(shared_case: Stage3SharedCase) -> Dict[str, Any]:
    out = asdict(shared_case)
    out["case_meta"]["x_center"] = float(out["case_meta"]["x_center"])
    out["case_meta"]["y_center"] = float(out["case_meta"]["y_center"])
    return out
