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


def build_stage2_case_payload():
    slices = []
    for scan_index in range(1, 13):
        full_ilm = [[float(x), 20.0 + 0.2 * scan_index + 0.02 * (x - 10) ** 2] for x in range(0, 20)]
        slices.append(
            {
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
        )
    return {
        "stage2_schema_version": "v1",
        "case_id": "CASE-CANONICAL",
        "patient_id": "PCAN",
        "laterality": "R",
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


class Stage3CanonicalMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_stage3_main_module()

    def setUp(self):
        baseline_row = {
            "Axial_Length": 24.0,
            "Diagnosis": "Dx",
            "Stage": "S1",
        }
        self.shared_case = build_stage3_shared_structure(build_stage2_case_payload(), baseline_row)
        self.final_cloud = build_legacy_cloud_from_shared(self.shared_case)
        self.aligned_cloud = self.module.align_to_bmo_bfp(self.final_cloud)

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


if __name__ == "__main__":
    unittest.main()
