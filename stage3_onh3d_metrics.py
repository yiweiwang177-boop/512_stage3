import math
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import numpy as np

from stage3_onh3d_contract import ONH3DCase, validate_onh3d_case


@dataclass
class ONH3DMetricConfig:
    """Configuration for continuous 3D ONH metric sampling and aggregation.

    `low10_mean` is defined using `ceil(0.1 * N)` on valid samples.
    3D-MRA is treated as a discrete integral approximation along the true 3D
    BMO ring. Sector systems are classification-only views; they must not
    define the true MRW/MRA measurement geometry.
    """

    ring_sample_count: int = 128
    low_fraction: float = 0.1
    low_count_mode: Literal["ceil"] = "ceil"
    sector_systems: tuple[int, ...] = (8, 4, 2)
    local_band_radius_mm: Optional[float] = None
    max_connection_length_mm: Optional[float] = None
    enforce_connection_validity: bool = True
    continuity_window: Optional[int] = None


@dataclass
class ONH3DLocalFrame:
    """Minimal local frame for the measurement plane at one ring sample.

    The frame varies continuously along the periodic BMO ring and is not a
    fallback to any legacy radial slice plane.
    """

    origin_3d: np.ndarray
    tangent_3d: np.ndarray
    reference_normal_3d: np.ndarray
    plane_x_axis_3d: np.ndarray
    plane_y_axis_3d: np.ndarray
    plane_normal_3d: np.ndarray


@dataclass
class ONH3DRingSample:
    """One equal-arc-length working sample on the true 3D BMO ring.

    Samples are discrete integration nodes only, not the ring definition
    itself. Orientation comes from the periodic ring parameterization and must
    not collapse back to legacy 12-radial slice semantics.
    """

    sample_idx: int
    arc_length_s_mm: float
    delta_s_mm: float
    bmo_point_3d: np.ndarray
    bmo_tangent_3d: np.ndarray
    theta_ref_deg: float
    sector_8: Optional[str]
    sector_4: Optional[str]
    sector_2: Optional[str]
    local_frame: ONH3DLocalFrame


@dataclass
class ONH3DRingGeometry:
    """Closed periodic 3D BMO ring represented by an ordered control polyline.

    Equal-arc-length samples are working discretizations only. The ring remains
    a closed periodic object and cannot be reduced to the old 12 radial
    direction reference.
    """

    control_points_3d: np.ndarray
    closed: bool
    cumulative_arc_length_mm: np.ndarray
    total_arc_length_mm: float
    segment_lengths_mm: np.ndarray = field(repr=False)

    def point_at_arclength(self, s_mm: float) -> np.ndarray:
        if not self.closed or self.total_arc_length_mm <= 0:
            raise ValueError("Ring geometry must be a closed ring with positive arc length")
        s_wrapped = float(s_mm) % float(self.total_arc_length_mm)
        idx = int(np.searchsorted(self.cumulative_arc_length_mm, s_wrapped, side="right") - 1)
        idx = max(0, min(idx, len(self.control_points_3d) - 1))
        seg_len = float(self.segment_lengths_mm[idx])
        p0 = self.control_points_3d[idx]
        p1 = self.control_points_3d[(idx + 1) % len(self.control_points_3d)]
        if seg_len <= 0:
            return p0.copy()
        alpha = (s_wrapped - self.cumulative_arc_length_mm[idx]) / seg_len
        return p0 + alpha * (p1 - p0)

    def resample_equal_arclength(self, sample_count: int) -> np.ndarray:
        return resample_ring_equal_arclength(self, sample_count)

    def tangent_at_index(self, points_3d: np.ndarray, idx: int) -> np.ndarray:
        return compute_local_tangent_periodic(points_3d, idx)


@dataclass
class ONH3DLocalMetric:
    """Per-sample 3D MRW/MRA metric tied to the local ring tangent.

    `mrw_um` is the shortest valid 3D BMO-to-ILM distance. `phi_deg` is the
    angle between the connection vector and the local BMO ring tangent.
    `mra_contrib_mm2 = MRW_i * Δs_i * sin(phi_i)`.
    """

    sample_idx: int
    is_valid: bool
    invalid_reason: Optional[str]
    bmo_point_3d: np.ndarray
    ilm_point_3d: Optional[np.ndarray]
    connection_vec_3d: Optional[np.ndarray]
    mrw_um: float
    phi_deg: float
    delta_s_mm: float
    mra_contrib_mm2: float
    theta_ref_deg: float
    sector_8: Optional[str]
    sector_4: Optional[str]
    sector_2: Optional[str]


@dataclass
class ONH3DSectorMetricSummary:
    system: int
    label: str
    mrw_mean_um: float
    mrw_low10_mean_um: float
    mra_sum_mm2: float
    valid_sample_count: int


@dataclass
class ONH3DMetricResult:
    config: ONH3DMetricConfig
    ring_geometry: ONH3DRingGeometry
    ring_samples: list[ONH3DRingSample]
    detail: list[ONH3DLocalMetric]
    sector_summary_8: dict[str, ONH3DSectorMetricSummary]
    sector_summary_4: dict[str, ONH3DSectorMetricSummary]
    sector_summary_2: dict[str, ONH3DSectorMetricSummary]
    MRW_global_mean_um: float
    MRW_global_min_um: float
    MRW_global_low10_mean_um: float
    MRA_global_sum_mm2: float
    valid_sample_count: int
    total_sample_count: int
    meta: Dict[str, Any] = field(default_factory=dict)
    qc: Dict[str, Any] = field(default_factory=dict)


def _normalize_vector(vec: Any, *, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        if fallback is not None:
            return np.asarray(fallback, dtype=float)
        raise ValueError("Expected a finite vector with shape (3,)")
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        if fallback is not None:
            return np.asarray(fallback, dtype=float)
        raise ValueError("Zero-length vector is not allowed")
    return arr / norm


def _finite_numeric_values(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _triangle_closest_point(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Return the closest point on triangle ABC to the query point."""

    ab = b - a
    ac = c - a
    ap = point - a

    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab

    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        bc = c - b
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * bc

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w


def _sector_names_from_theta(theta_ref_deg: float, case: ONH3DCase) -> tuple[Optional[str], Optional[str], Optional[str]]:
    if theta_ref_deg is None or not np.isfinite(theta_ref_deg):
        return None, None, None
    theta = float(theta_ref_deg) % 360.0
    sector_8 = None
    for idx, (lo, hi, label) in enumerate(case.sector_reference.sector_8_bounds):
        if idx == len(case.sector_reference.sector_8_bounds) - 1:
            if lo <= theta <= hi:
                sector_8 = label
                break
        elif lo <= theta < hi:
            sector_8 = label
            break
    if sector_8 is None:
        return None, None, None
    return (
        sector_8,
        case.sector_reference.sector_4_map.get(sector_8),
        case.sector_reference.sector_2_map.get(sector_8),
    )


def _compute_low_fraction_count(n: int, fraction: float, mode: str) -> int:
    if n <= 0:
        return 0
    if mode != "ceil":
        raise ValueError(f"Unsupported low_count_mode: {mode}")
    return max(1, int(math.ceil(float(fraction) * int(n))))


def build_ring_geometry(bmo_ring_3d: Any) -> ONH3DRingGeometry:
    points = np.asarray(bmo_ring_3d, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
        raise ValueError("bmo_ring_3d must have shape (N,3) with N>=3")
    if not np.all(np.isfinite(points)):
        raise ValueError("bmo_ring_3d must be finite")
    if np.allclose(points[0], points[-1]):
        points = points[:-1]
    if points.shape[0] < 3:
        raise ValueError("bmo_ring_3d must contain at least 3 unique control points")
    step_vecs = np.roll(points, -1, axis=0) - points
    step_lengths = np.linalg.norm(step_vecs, axis=1)
    if np.any(step_lengths <= 0):
        raise ValueError("bmo_ring_3d must not contain zero-length adjacent segments")
    cumulative = np.concatenate([[0.0], np.cumsum(step_lengths)])
    total = float(cumulative[-1])
    if total <= 0:
        raise ValueError("bmo_ring_3d total arc length must be positive")
    return ONH3DRingGeometry(
        control_points_3d=points,
        closed=True,
        cumulative_arc_length_mm=cumulative,
        total_arc_length_mm=total,
        segment_lengths_mm=step_lengths,
    )


def resample_ring_equal_arclength(ring_geometry: ONH3DRingGeometry, sample_count: int) -> np.ndarray:
    sample_count = int(sample_count)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    step = float(ring_geometry.total_arc_length_mm) / sample_count
    return np.vstack([ring_geometry.point_at_arclength(i * step) for i in range(sample_count)])


def compute_local_tangent_periodic(sample_points_3d: Any, idx: int) -> np.ndarray:
    points = np.asarray(sample_points_3d, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
        raise ValueError("sample_points_3d must have shape (N,3) with N>=3")
    n = points.shape[0]
    prev_pt = points[(idx - 1) % n]
    next_pt = points[(idx + 1) % n]
    tangent = next_pt - prev_pt
    norm = float(np.linalg.norm(tangent))
    if norm > 0:
        return tangent / norm
    forward = points[(idx + 1) % n] - points[idx]
    norm = float(np.linalg.norm(forward))
    if norm > 0:
        return forward / norm
    raise ValueError("Unable to compute periodic tangent from degenerate sample points")


def assign_sector_labels_from_reference(
    point_3d: Any,
    case: ONH3DCase,
    tangent_3d: Any = None,
) -> tuple[float, Optional[str], Optional[str], Optional[str]]:
    del tangent_3d  # Tangent is not used for labels; labels are reference-only.
    point = np.asarray(point_3d, dtype=float)
    vec = point - np.asarray(case.bmo_plane.origin_3d, dtype=float)
    x_axis = _normalize_vector(case.bmo_plane.x_axis_3d)
    y_axis = _normalize_vector(case.bmo_plane.y_axis_3d)
    zero_axis = _normalize_vector(case.sector_reference.angle_zero_axis_3d, fallback=x_axis)

    px = float(np.dot(vec, x_axis))
    py = float(np.dot(vec, y_axis))
    zx = float(np.dot(zero_axis, x_axis))
    zy = float(np.dot(zero_axis, y_axis))

    point_angle = math.degrees(math.atan2(py, px))
    zero_angle = math.degrees(math.atan2(zy, zx))
    theta = (point_angle - zero_angle) % 360.0
    if case.sector_reference.clockwise_positive:
        theta = (-theta) % 360.0
    sector_8, sector_4, sector_2 = _sector_names_from_theta(theta, case)
    return float(theta), sector_8, sector_4, sector_2


def build_ring_samples(
    case: ONH3DCase,
    config: ONH3DMetricConfig,
    ring_geometry: ONH3DRingGeometry,
) -> list[ONH3DRingSample]:
    """Build equal-arc-length ring samples and their local continuous frames.

    The local frame uses the BMO plane normal only as a reference direction for
    the measurement plane construction. It does not replace the true 3D BMO
    ring or define MRW/MRA geometry by itself.
    """

    sample_points = resample_ring_equal_arclength(ring_geometry, config.ring_sample_count)
    delta_s = float(ring_geometry.total_arc_length_mm) / int(config.ring_sample_count)
    ref_normal = _normalize_vector(case.bmo_plane.normal_3d)
    x_fallback = _normalize_vector(case.bmo_plane.x_axis_3d)
    ring_samples: list[ONH3DRingSample] = []

    for idx, point in enumerate(sample_points):
        tangent = compute_local_tangent_periodic(sample_points, idx)
        plane_normal = np.cross(tangent, ref_normal)
        if np.linalg.norm(plane_normal) <= 0:
            plane_normal = np.cross(tangent, x_fallback)
        plane_normal = _normalize_vector(plane_normal)
        plane_x = tangent
        plane_y = _normalize_vector(np.cross(plane_normal, plane_x))
        local_frame = ONH3DLocalFrame(
            origin_3d=np.asarray(point, dtype=float),
            tangent_3d=np.asarray(tangent, dtype=float),
            reference_normal_3d=np.asarray(ref_normal, dtype=float),
            plane_x_axis_3d=np.asarray(plane_x, dtype=float),
            plane_y_axis_3d=np.asarray(plane_y, dtype=float),
            plane_normal_3d=np.asarray(plane_normal, dtype=float),
        )
        theta_ref_deg, sector_8, sector_4, sector_2 = assign_sector_labels_from_reference(point, case, tangent)
        ring_samples.append(
            ONH3DRingSample(
                sample_idx=idx,
                arc_length_s_mm=float(idx * delta_s),
                delta_s_mm=delta_s,
                bmo_point_3d=np.asarray(point, dtype=float),
                bmo_tangent_3d=np.asarray(tangent, dtype=float),
                theta_ref_deg=theta_ref_deg,
                sector_8=sector_8,
                sector_4=sector_4,
                sector_2=sector_2,
                local_frame=local_frame,
            )
        )

    return ring_samples


def compute_low_fraction_mean(values: Any, fraction: float = 0.1, mode: str = "ceil") -> float:
    finite = np.sort(_finite_numeric_values(values))
    if finite.size == 0:
        return float("nan")
    k = _compute_low_fraction_count(finite.size, fraction, mode)
    return float(np.mean(finite[:k]))


def aggregate_global_metrics(detail_metrics: list[ONH3DLocalMetric], config: ONH3DMetricConfig) -> Dict[str, Any]:
    valid_mrw = [
        float(metric.mrw_um)
        for metric in detail_metrics
        if metric.is_valid and np.isfinite(metric.mrw_um)
    ]
    valid_mra = [
        float(metric.mra_contrib_mm2)
        for metric in detail_metrics
        if metric.is_valid and np.isfinite(metric.mra_contrib_mm2)
    ]
    return {
        "MRW_global_mean_um": float(np.mean(valid_mrw)) if valid_mrw else float("nan"),
        "MRW_global_min_um": float(np.min(valid_mrw)) if valid_mrw else float("nan"),
        "MRW_global_low10_mean_um": compute_low_fraction_mean(
            valid_mrw,
            fraction=config.low_fraction,
            mode=config.low_count_mode,
        ),
        "MRA_global_sum_mm2": float(np.sum(valid_mra)) if valid_mra else 0.0,
        "valid_sample_count": int(sum(1 for metric in detail_metrics if metric.is_valid)),
        "total_sample_count": int(len(detail_metrics)),
    }


def aggregate_sector_metrics(
    detail_metrics: list[ONH3DLocalMetric],
    config: ONH3DMetricConfig,
) -> Dict[int, Dict[str, ONH3DSectorMetricSummary]]:
    field_by_system = {8: "sector_8", 4: "sector_4", 2: "sector_2"}
    summaries: Dict[int, Dict[str, ONH3DSectorMetricSummary]] = {}
    for system in config.sector_systems:
        field_name = field_by_system.get(int(system))
        if field_name is None:
            continue
        grouped: Dict[str, list[ONH3DLocalMetric]] = {}
        for metric in detail_metrics:
            if not metric.is_valid:
                continue
            label = getattr(metric, field_name)
            if label is None:
                continue
            grouped.setdefault(str(label), []).append(metric)

        system_summary: Dict[str, ONH3DSectorMetricSummary] = {}
        for label, group in grouped.items():
            mrw_vals = [float(m.mrw_um) for m in group if np.isfinite(m.mrw_um)]
            mra_vals = [float(m.mra_contrib_mm2) for m in group if np.isfinite(m.mra_contrib_mm2)]
            system_summary[label] = ONH3DSectorMetricSummary(
                system=int(system),
                label=label,
                mrw_mean_um=float(np.mean(mrw_vals)) if mrw_vals else float("nan"),
                mrw_low10_mean_um=compute_low_fraction_mean(
                    mrw_vals,
                    fraction=config.low_fraction,
                    mode=config.low_count_mode,
                ),
                mra_sum_mm2=float(np.sum(mra_vals)) if mra_vals else 0.0,
                valid_sample_count=len(group),
            )
        summaries[int(system)] = system_summary

    return summaries


def select_valid_connection(
    sample: ONH3DRingSample,
    case: ONH3DCase,
    config: ONH3DMetricConfig,
) -> Optional[Dict[str, Any]]:
    """Select a valid local BMO-to-ILM connection candidate.

    V1 uses the canonical ILM triangle mesh directly. Candidate faces are
    limited to a narrow slab around the local measurement plane, the closest
    point on each candidate triangle is evaluated, and the shortest connection
    on the positive local measurement side is kept. Smoothness comes from the
    continuous ring/local-frame definition and the periodic sampling; this
    function only selects a valid local connection and never redefines
    `phi_deg = angle(connection_vec, tangent)`.
    """

    vertices = np.asarray(case.ilm_surface.vertices_3d, dtype=float)
    faces = np.asarray(case.ilm_surface.faces, dtype=int)
    origin = np.asarray(sample.local_frame.origin_3d, dtype=float)
    plane_normal = _normalize_vector(sample.local_frame.plane_normal_3d)
    plane_y = _normalize_vector(sample.local_frame.plane_y_axis_3d)

    band_radius = float(config.local_band_radius_mm) if config.local_band_radius_mm is not None else max(
        0.05,
        float(sample.delta_s_mm) * 0.5,
    )
    max_connection_length = (
        float(config.max_connection_length_mm)
        if config.max_connection_length_mm is not None
        else None
    )

    best: Optional[Dict[str, Any]] = None
    rejected_due_to_direction = 0
    rejected_due_to_length = 0
    rejected_due_to_band = 0

    for face in faces:
        tri = vertices[np.asarray(face, dtype=int)]
        signed_dist = np.dot(tri - origin, plane_normal)
        min_signed = float(np.min(signed_dist))
        max_signed = float(np.max(signed_dist))
        # The candidate face must intersect the local measurement-plane slab.
        if min_signed > band_radius or max_signed < -band_radius:
            continue

        closest = _triangle_closest_point(origin, tri[0], tri[1], tri[2])
        connection_vec = closest - origin
        length_mm = float(np.linalg.norm(connection_vec))
        if not np.isfinite(length_mm) or length_mm <= 0.0:
            continue

        plane_offset = abs(float(np.dot(connection_vec, plane_normal)))
        if plane_offset > band_radius:
            rejected_due_to_band += 1
            continue

        # Valid connections must remain on the positive measurement side of the
        # local frame. This filters back-facing or opposite-side candidates
        # without changing phi = angle(connection, tangent).
        if float(np.dot(connection_vec, plane_y)) <= 0.0:
            rejected_due_to_direction += 1
            continue

        if max_connection_length is not None and length_mm > max_connection_length:
            rejected_due_to_length += 1
            continue

        candidate = {
            "is_valid": True,
            "ilm_point_3d": closest,
            "connection_vec_3d": connection_vec,
            "connection_length_mm": length_mm,
            "plane_offset_mm": plane_offset,
        }
        if best is None:
            best = candidate
            continue
        if length_mm < float(best["connection_length_mm"]) - 1e-9:
            best = candidate
            continue
        if abs(length_mm - float(best["connection_length_mm"])) <= 1e-9 and plane_offset < float(best["plane_offset_mm"]):
            best = candidate

    if best is not None:
        return best

    if rejected_due_to_direction:
        reason = "no_candidate_on_positive_measurement_side"
    elif rejected_due_to_band:
        reason = "no_candidate_within_local_plane_band"
    elif rejected_due_to_length:
        reason = "no_candidate_within_max_connection_length"
    else:
        reason = "no_candidate_face_in_local_band"
    return {"is_valid": False, "invalid_reason": reason}


def compute_local_metric(
    sample: ONH3DRingSample,
    connection_result: Optional[Dict[str, Any]],
) -> ONH3DLocalMetric:
    if not connection_result or not bool(connection_result.get("is_valid", True)):
        return ONH3DLocalMetric(
            sample_idx=sample.sample_idx,
            is_valid=False,
            invalid_reason=connection_result.get("invalid_reason") if connection_result else "no_valid_connection",
            bmo_point_3d=np.asarray(sample.bmo_point_3d, dtype=float),
            ilm_point_3d=None,
            connection_vec_3d=None,
            mrw_um=float("nan"),
            phi_deg=float("nan"),
            delta_s_mm=float(sample.delta_s_mm),
            mra_contrib_mm2=0.0,
            theta_ref_deg=float(sample.theta_ref_deg),
            sector_8=sample.sector_8,
            sector_4=sample.sector_4,
            sector_2=sample.sector_2,
        )

    bmo_point = np.asarray(sample.bmo_point_3d, dtype=float)
    ilm_point = connection_result.get("ilm_point_3d")
    connection_vec = connection_result.get("connection_vec_3d")
    if ilm_point is None and connection_vec is None:
        raise ValueError("connection_result must provide ilm_point_3d or connection_vec_3d")

    if ilm_point is not None:
        ilm_point = np.asarray(ilm_point, dtype=float)
    if connection_vec is not None:
        connection_vec = np.asarray(connection_vec, dtype=float)

    if ilm_point is None:
        ilm_point = bmo_point + connection_vec
    if connection_vec is None:
        connection_vec = ilm_point - bmo_point

    mrw_mm = float(np.linalg.norm(connection_vec))
    tangent = _normalize_vector(sample.bmo_tangent_3d)
    if mrw_mm <= 0:
        raise ValueError("connection_vec_3d must have positive length for a valid metric")
    conn_unit = connection_vec / mrw_mm
    cos_phi = float(np.clip(np.dot(conn_unit, tangent), -1.0, 1.0))
    phi_deg = float(np.degrees(np.arccos(cos_phi)))
    mra_contrib_mm2 = float(mrw_mm * float(sample.delta_s_mm) * math.sin(math.radians(phi_deg)))

    return ONH3DLocalMetric(
        sample_idx=sample.sample_idx,
        is_valid=True,
        invalid_reason=None,
        bmo_point_3d=bmo_point,
        ilm_point_3d=ilm_point,
        connection_vec_3d=connection_vec,
        mrw_um=float(mrw_mm * 1000.0),
        phi_deg=phi_deg,
        delta_s_mm=float(sample.delta_s_mm),
        mra_contrib_mm2=mra_contrib_mm2,
        theta_ref_deg=float(sample.theta_ref_deg),
        sector_8=sample.sector_8,
        sector_4=sample.sector_4,
        sector_2=sample.sector_2,
    )


def compute_onh3d_metrics(
    case: ONH3DCase,
    config: Optional[ONH3DMetricConfig] = None,
) -> ONH3DMetricResult:
    """Compute the ONH3D metric skeleton from the true 3D BMO ring contract.

    The upstream `bmo_ring_3d` is not required to already contain 128 points.
    Fixed equal-arc-length working samples are generated inside the metrics
    core from the original ordered closed ring.
    """

    validate_onh3d_case(case)
    config = config or ONH3DMetricConfig()
    ring_geometry = build_ring_geometry(case.bmo_ring_3d)
    ring_samples = build_ring_samples(case, config, ring_geometry)

    detail_metrics = []
    for sample in ring_samples:
        connection_result = select_valid_connection(sample, case, config)
        detail_metrics.append(compute_local_metric(sample, connection_result))

    global_summary = aggregate_global_metrics(detail_metrics, config)
    sector_summary = aggregate_sector_metrics(detail_metrics, config)

    return ONH3DMetricResult(
        config=config,
        ring_geometry=ring_geometry,
        ring_samples=ring_samples,
        detail=detail_metrics,
        sector_summary_8=sector_summary.get(8, {}),
        sector_summary_4=sector_summary.get(4, {}),
        sector_summary_2=sector_summary.get(2, {}),
        MRW_global_mean_um=global_summary["MRW_global_mean_um"],
        MRW_global_min_um=global_summary["MRW_global_min_um"],
        MRW_global_low10_mean_um=global_summary["MRW_global_low10_mean_um"],
        MRA_global_sum_mm2=global_summary["MRA_global_sum_mm2"],
        valid_sample_count=global_summary["valid_sample_count"],
        total_sample_count=global_summary["total_sample_count"],
        meta={
            "algorithm_family": "ONH3D_512",
            "geometry_model": str(case.geometry_model),
            "algorithm_version": str(case.algorithm_version),
            "mra_sector_aggregation": "sum",
            "ring_sample_count": int(config.ring_sample_count),
        },
        qc={
            "ring_closed": bool(ring_geometry.closed),
            "total_arc_length_mm": float(ring_geometry.total_arc_length_mm),
            "valid_sample_count": int(global_summary["valid_sample_count"]),
            "total_sample_count": int(global_summary["total_sample_count"]),
        },
    )
