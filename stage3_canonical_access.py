from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from stage3_shared import NATIVE_SPACING_Z_MM, Stage3SharedCase, build_scale_x, pixel_to_3d


def get_shared_case(container: Dict[str, Any]) -> Optional[Stage3SharedCase]:
    shared_case = container.get("shared_case") if isinstance(container, dict) else None
    return shared_case if isinstance(shared_case, Stage3SharedCase) else None


def split_full_ilm_into_lr_from_canonical(slice_meta) -> Tuple[Optional[List[Tuple[float, float]]], Optional[List[Tuple[float, float]]]]:
    pts = sorted(slice_meta.full_ilm_px, key=lambda item: item[0])
    if len(pts) < 4:
        return None, None

    divider_x = None
    left_x = float(slice_meta.bmo_left_px[0])
    right_x = float(slice_meta.bmo_right_px[0])
    if np.isfinite(left_x) and np.isfinite(right_x):
        divider_x = 0.5 * (left_x + right_x)

    if divider_x is None or not np.isfinite(divider_x):
        divider_x = float(slice_meta.x_center)

    left_pts = [pt for pt in pts if pt[0] <= divider_x]
    right_pts = [pt for pt in pts if pt[0] >= divider_x]

    if len(left_pts) < 2 or len(right_pts) < 2:
        divider_x = float(slice_meta.x_center)
        left_pts = [pt for pt in pts if pt[0] <= divider_x]
        right_pts = [pt for pt in pts if pt[0] >= divider_x]

    if len(left_pts) < 2 or len(right_pts) < 2:
        return None, None

    return left_pts, right_pts


def _apply_alignment_to_point(point_3d: Tuple[float, float, float], alignment: Dict[str, Any]) -> Tuple[float, float, float]:
    pt = np.asarray(point_3d, dtype=float)
    centroid = np.asarray(alignment["centroid"], dtype=float)
    rotation_matrix = np.asarray(alignment["rotation_matrix"], dtype=float)
    aligned = rotation_matrix @ (pt - centroid)
    return tuple(aligned.tolist())


def _point_dict(slice_id: int, scan_index: int, side: str, pt2d, pt3d) -> Dict[str, Any]:
    return {
        "slice_id": int(slice_id),
        "scan_index": int(scan_index),
        "side": side,
        "pixel_x": float(pt2d[0]),
        "pixel_y": float(pt2d[1]),
        "point_3d": tuple(np.asarray(pt3d, dtype=float).tolist()),
    }


def _point_dict_no_side(slice_id: int, scan_index: int, pt2d, pt3d) -> Dict[str, Any]:
    return {
        "slice_id": int(slice_id),
        "scan_index": int(scan_index),
        "pixel_x": float(pt2d[0]),
        "pixel_y": float(pt2d[1]),
        "point_3d": tuple(np.asarray(pt3d, dtype=float).tolist()),
    }


def build_unaligned_canonical_slice_geometry(shared_case: Stage3SharedCase) -> Dict[int, Dict[str, Any]]:
    scale_x = build_scale_x(shared_case.case_meta.axial_length)
    scale_z = NATIVE_SPACING_Z_MM
    laterality = shared_case.case_meta.laterality

    out: Dict[int, Dict[str, Any]] = {}
    for slice_meta in shared_case.slice_meta_list:
        left_ilm_px, right_ilm_px = split_full_ilm_into_lr_from_canonical(slice_meta)

        bmo_lr = {
            "L": _point_dict(
                slice_meta.slice_id,
                slice_meta.scan_index,
                "L",
                slice_meta.bmo_left_px,
                pixel_to_3d(
                    slice_meta.bmo_left_px,
                    x_center=slice_meta.x_center,
                    scale_x=scale_x,
                    scale_z=scale_z,
                    angle_deg=slice_meta.angle_deg,
                    laterality=laterality,
                ),
            ),
            "R": _point_dict(
                slice_meta.slice_id,
                slice_meta.scan_index,
                "R",
                slice_meta.bmo_right_px,
                pixel_to_3d(
                    slice_meta.bmo_right_px,
                    x_center=slice_meta.x_center,
                    scale_x=scale_x,
                    scale_z=scale_z,
                    angle_deg=slice_meta.angle_deg,
                    laterality=laterality,
                ),
            ),
        }

        cutoff_lr = {
            "L": _point_dict(
                slice_meta.slice_id,
                slice_meta.scan_index,
                "L",
                slice_meta.cutoff_left_px,
                pixel_to_3d(
                    slice_meta.cutoff_left_px,
                    x_center=slice_meta.x_center,
                    scale_x=scale_x,
                    scale_z=scale_z,
                    angle_deg=slice_meta.angle_deg,
                    laterality=laterality,
                ),
            ),
            "R": _point_dict(
                slice_meta.slice_id,
                slice_meta.scan_index,
                "R",
                slice_meta.cutoff_right_px,
                pixel_to_3d(
                    slice_meta.cutoff_right_px,
                    x_center=slice_meta.x_center,
                    scale_x=scale_x,
                    scale_z=scale_z,
                    angle_deg=slice_meta.angle_deg,
                    laterality=laterality,
                ),
            ),
        }

        rnfl_effective_seg = np.array(
            [
                pixel_to_3d(
                    pt2d,
                    x_center=slice_meta.x_center,
                    scale_x=scale_x,
                    scale_z=scale_z,
                    angle_deg=slice_meta.angle_deg,
                    laterality=laterality,
                )
                for pt2d in slice_meta.rnfl_effective_lower_px
            ],
            dtype=float,
        )

        ilm_lr = {}
        if left_ilm_px is not None and right_ilm_px is not None:
            ilm_lr["L"] = np.array(
                [
                    pixel_to_3d(
                        pt2d,
                        x_center=slice_meta.x_center,
                        scale_x=scale_x,
                        scale_z=scale_z,
                        angle_deg=slice_meta.angle_deg,
                        laterality=laterality,
                    )
                    for pt2d in left_ilm_px
                ],
                dtype=float,
            )
            ilm_lr["R"] = np.array(
                [
                    pixel_to_3d(
                        pt2d,
                        x_center=slice_meta.x_center,
                        scale_x=scale_x,
                        scale_z=scale_z,
                        angle_deg=slice_meta.angle_deg,
                        laterality=laterality,
                    )
                    for pt2d in right_ilm_px
                ],
                dtype=float,
            )

        out[int(slice_meta.slice_id)] = {
            "slice_meta": slice_meta,
            "bmo_lr": bmo_lr,
            "cutoff_lr": cutoff_lr,
            "rnfl_effective_seg": rnfl_effective_seg,
            "ilm_lr": ilm_lr,
            "full_ilm_points": [
                _point_dict_no_side(
                    slice_meta.slice_id,
                    slice_meta.scan_index,
                    pt2d,
                    pixel_to_3d(
                        pt2d,
                        x_center=slice_meta.x_center,
                        scale_x=scale_x,
                        scale_z=scale_z,
                        angle_deg=slice_meta.angle_deg,
                        laterality=laterality,
                    ),
                )
                for pt2d in slice_meta.full_ilm_px
            ],
        }

    return out


def build_aligned_canonical_slice_geometry(
    shared_case: Stage3SharedCase,
    alignment: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    unaligned = build_unaligned_canonical_slice_geometry(shared_case)
    out: Dict[int, Dict[str, Any]] = {}

    for sid, geom in unaligned.items():
        bmo_lr = {}
        for side, item in geom["bmo_lr"].items():
            bmo_lr[side] = dict(item)
            bmo_lr[side]["point_3d"] = _apply_alignment_to_point(item["point_3d"], alignment)

        cutoff_lr = {}
        for side, item in geom["cutoff_lr"].items():
            cutoff_lr[side] = dict(item)
            cutoff_lr[side]["point_3d"] = _apply_alignment_to_point(item["point_3d"], alignment)

        ilm_lr = {}
        for side, poly in geom["ilm_lr"].items():
            ilm_lr[side] = np.array(
                [_apply_alignment_to_point(tuple(pt.tolist()), alignment) for pt in np.asarray(poly, dtype=float)],
                dtype=float,
            )

        rnfl_effective_seg = np.array(
            [_apply_alignment_to_point(tuple(pt.tolist()), alignment) for pt in np.asarray(geom["rnfl_effective_seg"], dtype=float)],
            dtype=float,
        )

        full_ilm_points = []
        for item in geom["full_ilm_points"]:
            aligned_item = dict(item)
            aligned_item["point_3d"] = _apply_alignment_to_point(item["point_3d"], alignment)
            full_ilm_points.append(aligned_item)

        out[sid] = {
            "slice_meta": geom["slice_meta"],
            "bmo_lr": bmo_lr,
            "cutoff_lr": cutoff_lr,
            "rnfl_effective_seg": rnfl_effective_seg,
            "ilm_lr": ilm_lr,
            "full_ilm_points": full_ilm_points,
        }

    return out


def build_ordered_bmo24_from_canonical_slice_geometry(
    canonical_slice_geometry: Dict[int, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    ordered = []
    for sid in sorted(canonical_slice_geometry.keys()):
        ordered.append(
            {
                "slice_id": sid,
                "side": "L",
                "point_3d": np.array(canonical_slice_geometry[sid]["bmo_lr"]["L"]["point_3d"], dtype=float),
            }
        )
    for sid in sorted(canonical_slice_geometry.keys()):
        ordered.append(
            {
                "slice_id": sid,
                "side": "R",
                "point_3d": np.array(canonical_slice_geometry[sid]["bmo_lr"]["R"]["point_3d"], dtype=float),
            }
        )
    return ordered
