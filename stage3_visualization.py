import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from stage3_canonical_access import (
    build_aligned_canonical_slice_geometry,
    build_ordered_bmo24_from_canonical_slice_geometry,
    get_shared_case,
)


def _collect_geometry_points(canonical_geom: Dict[int, Dict[str, Any]]) -> List[np.ndarray]:
    points: List[np.ndarray] = []
    for sid in sorted(canonical_geom.keys()):
        geom = canonical_geom[sid]
        for side in ["L", "R"]:
            points.append(np.asarray(geom["bmo_lr"][side]["point_3d"], dtype=float))
        rnfl = np.asarray(geom["rnfl_effective_seg"], dtype=float)
        if rnfl.ndim == 2 and len(rnfl) > 0:
            points.extend(rnfl)
        for side in ["L", "R"]:
            ilm = np.asarray(geom["ilm_lr"].get(side, []), dtype=float)
            if ilm.ndim == 2 and len(ilm) > 0:
                points.extend(ilm)
    return points


def _set_axes_equal(ax, points: List[np.ndarray]) -> None:
    if not points:
        return
    stacked = np.vstack(points)
    mins = np.nanmin(stacked, axis=0)
    maxs = np.nanmax(stacked, axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.nanmax(maxs - mins)) / 2.0, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _plot_polyline(ax, pts, *, color, linewidth=1.2, alpha=1.0, label=None):
    arr = np.asarray(pts, dtype=float)
    if arr.ndim != 2 or len(arr) < 2:
        return
    ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color=color, linewidth=linewidth, alpha=alpha, label=label)


def _render_single_view(canonical_geom, laterality, output_path, *, elev, azim, title_suffix):
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    ordered_bmo = build_ordered_bmo24_from_canonical_slice_geometry(canonical_geom)
    bmo_ring = [np.asarray(item["point_3d"], dtype=float) for item in ordered_bmo]
    if bmo_ring:
        bmo_ring.append(bmo_ring[0])
        _plot_polyline(ax, bmo_ring, color="#00c4ff", linewidth=2.4, alpha=0.95, label="BMO ring")

    added_labels = set()
    for sid in sorted(canonical_geom.keys()):
        geom = canonical_geom[sid]
        left = np.asarray(geom["bmo_lr"]["L"]["point_3d"], dtype=float)
        right = np.asarray(geom["bmo_lr"]["R"]["point_3d"], dtype=float)
        slice_label = "Radial slice" if "Radial slice" not in added_labels else None
        _plot_polyline(ax, [left, right], color="#7a7a7a", linewidth=1.1, alpha=0.9, label=slice_label)
        added_labels.add("Radial slice")

        rnfl = np.asarray(geom["rnfl_effective_seg"], dtype=float)
        rnfl_label = "Effective RNFL lower" if "Effective RNFL lower" not in added_labels else None
        _plot_polyline(ax, rnfl, color="#19b34a", linewidth=1.6, alpha=0.9, label=rnfl_label)
        added_labels.add("Effective RNFL lower")

        for side, color in [("L", "#f4b400"), ("R", "#db4437")]:
            ilm = np.asarray(geom["ilm_lr"].get(side, []), dtype=float)
            ilm_label = "ILM" if "ILM" not in added_labels else None
            _plot_polyline(ax, ilm, color=color, linewidth=1.0, alpha=0.8, label=ilm_label)
            added_labels.add("ILM")

        midpoint = 0.5 * (left + right)
        ax.text(midpoint[0], midpoint[1], midpoint[2], str(sid), fontsize=7, color="#333333")

    all_points = _collect_geometry_points(canonical_geom)
    _set_axes_equal(ax, all_points)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"Stage3 3D QC | {title_suffix} | Laterality={laterality}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_stage3_qc_3d_views(output_dir: str, aligned_cloud: Dict[str, Any]) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)

    shared_case = get_shared_case(aligned_cloud)
    alignment = aligned_cloud.get("ALIGNMENT") if isinstance(aligned_cloud, dict) else None
    if shared_case is None or alignment is None:
        return []

    canonical_geom = build_aligned_canonical_slice_geometry(shared_case, alignment)
    if not canonical_geom:
        return []

    laterality = shared_case.case_meta.laterality
    files = [
        ("stage3_qc_3d_top.png", 90.0, -90.0, "Top view"),
        ("stage3_qc_3d_oblique.png", 28.0, -58.0, "Oblique view"),
        ("stage3_qc_3d_side.png", 10.0, 0.0, "Side view"),
    ]

    written = []
    for filename, elev, azim, title_suffix in files:
        path = os.path.join(output_dir, filename)
        _render_single_view(
            canonical_geom,
            laterality,
            path,
            elev=elev,
            azim=azim,
            title_suffix=title_suffix,
        )
        written.append(path)

    manifest_path = os.path.join(output_dir, "qc_3d_manifest.json")
    manifest = {
        "laterality": laterality,
        "views": [
            {
                "file": os.path.basename(path),
                "purpose": purpose,
            }
            for path, purpose in [
                (written[0], "Top view for BMO ring shape and radial direction sanity"),
                (written[1], "Oblique view for overall aligned 3D geometry sanity"),
                (written[2], "Side view for ILM and effective RNFL lower vertical relationship"),
            ]
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    written.append(manifest_path)
    return written
