from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from stage3_reporting import SECTOR_2_MAP, SECTOR_4_MAP, SECTOR_8_BOUNDS


@dataclass
class BMOPlane:
    origin_3d: Any
    normal_3d: Any
    x_axis_3d: Any
    y_axis_3d: Any


@dataclass
class SectorReference:
    angle_zero_axis_3d: Any
    clockwise_positive: bool = True
    sector_8_bounds: Sequence[Tuple[float, float, str]] = field(default_factory=lambda: list(SECTOR_8_BOUNDS))
    sector_4_map: Dict[str, str] = field(default_factory=lambda: dict(SECTOR_4_MAP))
    sector_2_map: Dict[str, str] = field(default_factory=lambda: dict(SECTOR_2_MAP))


@dataclass
class TransformInfo:
    world_unit: str = "mm"
    voxel_spacing_mm_xyz: Optional[Tuple[float, float, float]] = None
    voxel_to_world: Any = None


@dataclass
class ILMSurfaceModel:
    vertices_3d: Any
    faces: Any
    vertex_normals_3d: Any = None
    surface_bounds_3d: Any = None


@dataclass
class ONH3DQCMeta:
    bmo_ring_closed: bool
    bmo_ring_sampling_count: int
    ilm_surface_vertex_count: int
    ilm_surface_face_count: int
    bmo_plane_fit_rms_mm: Optional[float] = None
    notes: Sequence[str] = field(default_factory=list)


@dataclass
class ONH3DCase:
    """Canonical 512-volume ONH 3D contract.

    `bmo_ring_3d` is the upstream-provided ordered 3D closed-ring representation.
    The metrics core is responsible for arc-length resampling it to the fixed
    working density, currently 128 samples. The contract does not require the
    input ring to already contain 128 points.
    """

    case_id: str
    patient_id: str
    laterality: str
    axial_length: float
    algorithm_version: str
    geometry_model: str
    bmo_ring_3d: Any
    bmo_plane: BMOPlane
    sector_reference: SectorReference
    transform_info: TransformInfo
    ilm_surface: ILMSurfaceModel
    qc_meta: ONH3DQCMeta
    diagnosis: Optional[str] = None
    stage: Optional[str] = None
    source_meta: Optional[Dict[str, Any]] = None


def _require_non_empty_text(value: Any, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _coerce_vector3(value: Any, field_name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{field_name} must have shape (3,)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{field_name} must be finite")
    return arr


def _coerce_points_n3(value: Any, field_name: str, *, min_rows: int = 1) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < min_rows:
        raise ValueError(f"{field_name} must have shape (N,3) with N>={min_rows}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{field_name} must be finite")
    return arr


def _coerce_faces_k3(value: Any, field_name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 1:
        raise ValueError(f"{field_name} must have shape (K,3) with K>=1")
    try:
        arr = arr.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be integer-convertible") from exc
    return arr


def validate_onh3d_case(case: ONH3DCase) -> None:
    _require_non_empty_text(case.case_id, "case_id")
    _require_non_empty_text(case.patient_id, "patient_id")
    laterality = _require_non_empty_text(case.laterality, "laterality").upper()
    if laterality not in {"L", "R"}:
        raise ValueError("laterality must be 'L' or 'R'")
    _require_non_empty_text(case.algorithm_version, "algorithm_version")
    _require_non_empty_text(case.geometry_model, "geometry_model")

    try:
        axial_length = float(case.axial_length)
    except (TypeError, ValueError) as exc:
        raise ValueError("axial_length must be numeric") from exc
    if not np.isfinite(axial_length):
        raise ValueError("axial_length must be finite")

    _coerce_points_n3(case.bmo_ring_3d, "bmo_ring_3d", min_rows=3)

    _coerce_vector3(case.bmo_plane.origin_3d, "bmo_plane.origin_3d")
    _coerce_vector3(case.bmo_plane.normal_3d, "bmo_plane.normal_3d")
    _coerce_vector3(case.bmo_plane.x_axis_3d, "bmo_plane.x_axis_3d")
    _coerce_vector3(case.bmo_plane.y_axis_3d, "bmo_plane.y_axis_3d")

    _coerce_vector3(case.sector_reference.angle_zero_axis_3d, "sector_reference.angle_zero_axis_3d")

    _coerce_points_n3(case.ilm_surface.vertices_3d, "ilm_surface.vertices_3d", min_rows=3)
    _coerce_faces_k3(case.ilm_surface.faces, "ilm_surface.faces")

    if case.transform_info.voxel_spacing_mm_xyz is not None:
        spacing = np.asarray(case.transform_info.voxel_spacing_mm_xyz, dtype=float)
        if spacing.shape != (3,) or not np.all(np.isfinite(spacing)):
            raise ValueError("transform_info.voxel_spacing_mm_xyz must have shape (3,) and be finite")

    if case.transform_info.voxel_to_world is not None:
        matrix = np.asarray(case.transform_info.voxel_to_world, dtype=float)
        if matrix.ndim != 2 or matrix.shape not in {(4, 4), (3, 4), (3, 3)}:
            raise ValueError("transform_info.voxel_to_world must be a 3x3, 3x4, or 4x4 matrix")

    if case.ilm_surface.vertex_normals_3d is not None:
        _coerce_points_n3(case.ilm_surface.vertex_normals_3d, "ilm_surface.vertex_normals_3d", min_rows=1)

    if case.ilm_surface.surface_bounds_3d is not None:
        bounds = np.asarray(case.ilm_surface.surface_bounds_3d, dtype=float)
        if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
            raise ValueError("ilm_surface.surface_bounds_3d must have shape (2,3)")
