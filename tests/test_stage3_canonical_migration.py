import copy
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from stage3_canonical_access import (
    build_aligned_canonical_slice_geometry,
    build_unaligned_canonical_slice_geometry,
    get_shared_case,
)
from stage3_shared import build_legacy_cloud_from_shared, build_stage3_shared_structure


def build_stage2_case_payload(*, laterality="R", slice_overrides=None):
    slice_overrides = slice_overrides or {}
    slices = []
    for scan_index in range(1, 13):
        full_ilm = [
            [float(x), 20.0 + 0.2 * scan_index + 0.02 * (x - 10) ** 2]
            for x in range(0, 20)
        ]
        record = {
            "stage2_schema_version": "v1",
            "scan_index": scan_index,
            "slice_stem": f"slice_{scan_index:02d}",
            "image_width": 20,
            "image_height": 100,
            "full_ilm_px": full_ilm,
            "bmo_px": [[5.0, 40.0], [14.0, 40.0]],
            "cutoff_px": [[4.0, 34.0], [15.0, 34.0]],
            "rnfl_effective_lower_px": [[4.0, 60.0], [9.0, 62.0], [15.0, 60.0]],
            "review_status": "approved",
            "source_flags": {"stage2_final": True},
        }
        record.update(copy.deepcopy(slice_overrides.get(scan_index, {})))
        slices.append(record)
    return {
        "stage2_schema_version": "v1",
        "case_id": "CASE-CANONICAL",
        "patient_id": "PCAN",
        "laterality": laterality,
        "slices": slices,
    }


def load_stage3_main_module():
    root = Path(__file__).resolve().parents[1]
    candidates = [
        path
        for path in root.glob("*.py")
        if path.name not in {"stage3_input_adapter.py", "stage3_shared.py", "stage3_canonical_access.py"}
        and "pre" not in path.name.lower()
    ]
    assert len(candidates) == 1
    spec = importlib.util.spec_from_file_location("stage3_main_module", str(candidates[0]))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def poison_meta(points, *, with_side=False):
    poisoned = []
    for sid in range(1, 13):
        base = {"slice_id": sid, "scan_index": sid, "pixel_x": 999.0, "pixel_y": 999.0, "point_3d": (999.0, 999.0, 999.0)}
        if with_side:
            left_item = dict(base)
            left_item["side"] = "L"
            right_item = dict(base)
            right_item["side"] = "R"
            poisoned.extend([left_item, right_item])
        else:
            poisoned.extend([dict(base) for _ in range(points)])
    return poisoned


def build_case_artifacts(module, payload):
    baseline_row = {
        "Axial_Length": 24.0,
        "Diagnosis": "Dx",
        "Stage": "S1",
    }
    shared_case = build_stage3_shared_structure(payload, baseline_row)
    final_cloud = build_legacy_cloud_from_shared(shared_case)
    aligned_cloud = module.align_to_bmo_bfp(final_cloud)
    return shared_case, final_cloud, aligned_cloud


class Stage3CanonicalMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_stage3_main_module()

    def setUp(self):
        self.shared_case, self.final_cloud, self.aligned_cloud = build_case_artifacts(
            self.module,
            build_stage2_case_payload(),
        )

    def test_aligned_cloud_preserves_shared_case_for_canonical_access(self):
        self.assertIs(self.aligned_cloud["shared_case"], self.shared_case)
        self.assertIs(get_shared_case(self.aligned_cloud), self.shared_case)

    def test_scan_index_and_angle_preserved_after_canonical_access(self):
        unaligned = build_unaligned_canonical_slice_geometry(self.shared_case)
        aligned = build_aligned_canonical_slice_geometry(self.shared_case, self.aligned_cloud["ALIGNMENT"])

        self.assertEqual(unaligned[1]["slice_meta"].scan_index, 1)
        self.assertEqual(unaligned[1]["slice_meta"].angle_deg, 0.0)
        self.assertEqual(aligned[12]["slice_meta"].scan_index, 12)
        self.assertEqual(aligned[12]["slice_meta"].angle_deg, 165.0)

    def test_mrw_can_read_from_canonical_shared_case_without_ilm_meta_primary(self):
        clean = self.module.extract_mrw_segments_from_cloud(self.final_cloud)
        poisoned_cloud = copy.deepcopy(self.final_cloud)
        poisoned_cloud["ILM_META"] = poison_meta(points=4, with_side=False)
        poisoned_cloud["BMO_META"] = poison_meta(points=1, with_side=True)
        poisoned = self.module.extract_mrw_segments_from_cloud(poisoned_cloud)

        clean_pairs = [(item["slice_id"], item["side"], round(item["mrw_len"], 6)) for item in clean]
        poisoned_pairs = [(item["slice_id"], item["side"], round(item["mrw_len"], 6)) for item in poisoned]
        self.assertEqual(clean_pairs, poisoned_pairs)

    def test_mrw_uses_full_ilm_semantics_with_left_right_constraint(self):
        payload = build_stage2_case_payload(
            slice_overrides={
                1: {
                    "full_ilm_px": (
                        [[float(x), 12.0] for x in range(0, 10)] +
                        [[float(x), 39.0] for x in range(10, 20)]
                    )
                }
            }
        )
        _, final_cloud, _ = build_case_artifacts(self.module, payload)
        mrw = self.module.extract_mrw_segments_from_cloud(final_cloud)
        slice_meta = final_cloud["SLICE_META"][1]

        left = next(item for item in mrw if item["slice_id"] == 1 and item["side"] == "L")
        right = next(item for item in mrw if item["slice_id"] == 1 and item["side"] == "R")

        left_px = self.module.original_3d_to_image_px(np.asarray(left["ilm_pt"], dtype=float), slice_meta, final_cloud["laterality"])
        right_px = self.module.original_3d_to_image_px(np.asarray(right["ilm_pt"], dtype=float), slice_meta, final_cloud["laterality"])

        self.assertIsNotNone(left_px)
        self.assertIsNotNone(right_px)
        self.assertLessEqual(float(left_px[0]), 9.5)
        self.assertGreaterEqual(float(right_px[0]), 9.5)
        self.assertGreater(float(left["mrw_len"]), float(right["mrw_len"]))

    def test_mra_can_read_from_canonical_shared_case_without_ilm_meta_primary(self):
        clean_local, clean_global = self.module.calculate_gardiner_mra(self.aligned_cloud, n_sectors=24, phi_step_deg=1.0)
        poisoned_cloud = copy.deepcopy(self.aligned_cloud)
        poisoned_cloud["BMO_META"] = poison_meta(points=1, with_side=True)
        poisoned_cloud["ILM_META"] = poison_meta(points=4, with_side=False)
        poisoned_cloud["BMO"] = [[999.0, 999.0, 999.0] for _ in range(24)]
        poisoned_local, poisoned_global = self.module.calculate_gardiner_mra(poisoned_cloud, n_sectors=24, phi_step_deg=1.0)

        self.assertEqual(len(clean_local), len(poisoned_local))
        clean_summary = [
            (item["slice_id"], item["side"], round(item["local_area_mm2"], 6), round(item["rw_phi_um"], 6))
            for item in clean_local
        ]
        poisoned_summary = [
            (item["slice_id"], item["side"], round(item["local_area_mm2"], 6), round(item["rw_phi_um"], 6))
            for item in poisoned_local
        ]
        self.assertEqual(clean_summary, poisoned_summary)
        self.assertAlmostEqual(clean_global, poisoned_global, places=6)

    def test_mra_nearest_intersection_ignores_far_hit(self):
        polyline = np.array(
            [
                [-1.0, 0.0, -1.0],
                [1.0, 0.0, -1.0],
                [1.0, 0.0, -3.0],
                [-1.0, 0.0, -3.0],
            ],
            dtype=float,
        )
        rw_mm, hit_pt, hit_meta = self.module.intersect_ray_with_polyline_in_slice_2d(
            np.array([0.0, 0.0, 0.0], dtype=float),
            np.array([1.0, 0.0], dtype=float),
            polyline,
            0.0,
        )
        self.assertAlmostEqual(rw_mm, 1.0, places=6)
        np.testing.assert_allclose(hit_pt, np.array([0.0, 0.0, -1.0], dtype=float), atol=1e-6)
        self.assertEqual(hit_meta["segment_index"], 0)

    def test_mra_ignores_near_zero_self_hits(self):
        polyline = np.array(
            [
                [-1.0, 0.0, -0.00005],
                [1.0, 0.0, -0.00005],
                [1.0, 0.0, -3.0],
                [-1.0, 0.0, -3.0],
            ],
            dtype=float,
        )
        rw_mm, hit_pt, hit_meta = self.module.intersect_ray_with_polyline_in_slice_2d(
            np.array([0.0, 0.0, 0.0], dtype=float),
            np.array([1.0, 0.0], dtype=float),
            polyline,
            0.0,
        )
        self.assertAlmostEqual(rw_mm, 3.0, places=6)
        np.testing.assert_allclose(hit_pt, np.array([0.0, 0.0, -3.0], dtype=float), atol=1e-6)
        self.assertEqual(hit_meta["segment_index"], 2)

    def test_mra_signed_angle_search_covers_negative_and_positive(self):
        neg_phi = self.module.compute_mra_phi_signed_deg(
            self.module.build_mra_ray_dir_local(-30.0),
            np.array([-1.0, 1.0], dtype=float),
            np.array([1.0, 1.0], dtype=float),
        )
        pos_phi = self.module.compute_mra_phi_signed_deg(
            self.module.build_mra_ray_dir_local(30.0),
            np.array([-1.0, 1.0], dtype=float),
            np.array([1.0, 1.0], dtype=float),
        )
        self.assertLess(neg_phi, 0.0)
        self.assertGreater(pos_phi, 0.0)

    def test_mra_symmetric_geometry_preserves_area_magnitude(self):
        ray_neg = self.module.build_mra_ray_dir_local(-30.0)
        ray_pos = self.module.build_mra_ray_dir_local(30.0)
        phi_neg = self.module.compute_mra_phi_signed_deg(
            ray_neg,
            np.array([-1.0, 1.0], dtype=float),
            np.array([1.0, 1.0], dtype=float),
        )
        phi_pos = self.module.compute_mra_phi_signed_deg(
            ray_pos,
            np.array([-1.0, 1.0], dtype=float),
            np.array([1.0, 1.0], dtype=float),
        )
        self.assertAlmostEqual(phi_neg, -phi_pos, places=6)

        r_mm = 2.5
        bottom_len_mm = 0.8
        rw_phi_mm = 0.9
        phi_neg_abs = abs(phi_neg)
        phi_pos_abs = abs(phi_pos)
        self.assertAlmostEqual(phi_neg_abs, phi_pos_abs, places=6)

        top_neg = bottom_len_mm * ((r_mm - rw_phi_mm * np.cos(np.deg2rad(phi_neg_abs))) / r_mm)
        top_pos = bottom_len_mm * ((r_mm - rw_phi_mm * np.cos(np.deg2rad(phi_pos_abs))) / r_mm)
        area_neg = 0.5 * (bottom_len_mm + top_neg) * rw_phi_mm
        area_pos = 0.5 * (bottom_len_mm + top_pos) * rw_phi_mm
        self.assertAlmostEqual(area_neg, area_pos, places=6)

    def test_global_mra_mm2_is_sum_of_local_areas(self):
        local_rows, global_mra = self.module.calculate_gardiner_mra(
            self.aligned_cloud,
            n_sectors=24,
            phi_step_deg=1.0,
        )
        self.assertAlmostEqual(
            global_mra,
            sum(float(item["local_area_mm2"]) for item in local_rows),
            places=6,
        )

    def test_lcd_lcci_can_read_from_canonical_shared_case_without_ali_alcs_meta_primary(self):
        clean_df, _ = self.module.compute_traditional_lcd_lcci_all_slices(self.aligned_cloud)
        poisoned_cloud = copy.deepcopy(self.aligned_cloud)
        poisoned_cloud["BMO_META"] = poison_meta(points=1, with_side=True)
        poisoned_cloud["ALI_META"] = poison_meta(points=1, with_side=True)
        poisoned_cloud["ALCS_META"] = poison_meta(points=3, with_side=False)
        poisoned_df, _ = self.module.compute_traditional_lcd_lcci_all_slices(poisoned_cloud)

        cols = [
            "slice_id",
            "status",
            "Length_D_mm",
            "Area_S_mm2",
            "LCD_area_mm",
            "LCCI_area_mm",
        ]
        pd.testing.assert_frame_equal(
            clean_df[cols].round(6).reset_index(drop=True),
            poisoned_df[cols].round(6).reset_index(drop=True),
        )

    def test_lcd_lcci_uses_cutoff_as_ali_semantics(self):
        canonical_geom = build_aligned_canonical_slice_geometry(
            self.shared_case,
            self.aligned_cloud["ALIGNMENT"],
        )
        _, payload_map = self.module.compute_traditional_lcd_lcci_all_slices(self.aligned_cloud)
        payload = payload_map[1]

        expected = np.vstack(
            [
                canonical_geom[1]["cutoff_lr"]["L"]["point_3d"],
                canonical_geom[1]["cutoff_lr"]["R"]["point_3d"],
            ]
        )
        np.testing.assert_allclose(payload["ali_lr_xyz"], expected, atol=1e-6)

    def test_lcd_lcci_uses_effective_rnfl_seg_as_alcs_semantics(self):
        canonical_geom = build_aligned_canonical_slice_geometry(
            self.shared_case,
            self.aligned_cloud["ALIGNMENT"],
        )
        _, payload_map = self.module.compute_traditional_lcd_lcci_all_slices(self.aligned_cloud)
        payload = payload_map[1]

        np.testing.assert_allclose(
            payload["alcs_xyz"],
            canonical_geom[1]["rnfl_effective_seg"],
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
