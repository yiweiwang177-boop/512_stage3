import json
import shutil
import unittest
import uuid
from pathlib import Path

import pandas as pd

from stage3_input_adapter import load_patient_baseline_row, load_stage2_case
from stage3_shared import build_legacy_cloud_from_shared, build_stage3_shared_structure


def build_stage2_case_payload():
    slices = []
    for scan_index in range(1, 13):
        full_ilm = [[float(x), 20.0] for x in range(0, 12)]
        slices.append(
            {
                "scan_index": scan_index,
                "slice_stem": f"slice_{scan_index:02d}",
                "image_width": 12,
                "image_height": 80,
                "full_ilm_px": full_ilm,
                "bmo_px": [[3.0, 35.0], [8.0, 35.0]],
                "cutoff_px": [[2.0, 30.0], [9.0, 30.0]],
                "rnfl_effective_lower_px": [[2.0, 55.0], [5.0, 56.0], [9.0, 55.0]],
                "review_status": "approved",
                "source_flags": {"stage2_final": True},
                "image_path": None,
            }
        )
    return {
        "stage2_schema_version": "v1",
        "case_id": "CASE-LEGACY",
        "patient_id": "P100",
        "laterality": "R",
        "slices": slices,
    }


class Stage3LegacyCompatTests(unittest.TestCase):
    def setUp(self):
        workspace_tmp = Path(__file__).resolve().parents[1] / "tests_tmp"
        workspace_tmp.mkdir(exist_ok=True)
        self.root = workspace_tmp / f"legacy_compat_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.stage2_json = self.root / "case.json"
        self.base_table = self.root / "baseline.xlsx"

        with open(self.stage2_json, "w", encoding="utf-8") as handle:
            json.dump(build_stage2_case_payload(), handle, ensure_ascii=False, indent=2)

        pd.DataFrame(
            [
                {
                    "Patient_ID": "P100",
                    "Laterality": "R",
                    "Axial_Length": 24.2,
                }
            ]
        ).to_excel(self.base_table, index=False)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_build_legacy_cloud_from_shared_emits_legacy_keys(self):
        stage2_case = load_stage2_case(str(self.stage2_json), expected_case_id="CASE-LEGACY")
        baseline_row = load_patient_baseline_row(str(self.base_table), "P100", "R")
        shared_case = build_stage3_shared_structure(stage2_case, baseline_row)
        cloud = build_legacy_cloud_from_shared(shared_case)

        for key in [
            "BMO",
            "ALI",
            "ALCS",
            "ILM_ROI",
            "BMO_META",
            "ILM_META",
            "ALI_META",
            "ALCS_META",
            "SLICE_META",
            "laterality",
            "axial_length",
        ]:
            self.assertIn(key, cloud)

        self.assertEqual(len(cloud["SLICE_META"]), 12)
        self.assertEqual(cloud["laterality"], "R")

    def test_slice_meta_contains_image_shape_center_and_scan_index(self):
        stage2_case = load_stage2_case(str(self.stage2_json), expected_case_id="CASE-LEGACY")
        baseline_row = load_patient_baseline_row(str(self.base_table), "P100", "R")
        shared_case = build_stage3_shared_structure(stage2_case, baseline_row)
        cloud = build_legacy_cloud_from_shared(shared_case)

        first_slice = cloud["SLICE_META"][1]
        self.assertEqual(first_slice["scan_index"], 1)
        self.assertEqual(first_slice["image_shape"], (80, 12))
        self.assertEqual(first_slice["image_width"], 12)
        self.assertEqual(first_slice["image_height"], 80)
        self.assertEqual(first_slice["x_center"], 6.0)
        self.assertEqual(first_slice["y_center"], 40.0)
        self.assertEqual(first_slice["rotation_center_px"], [6.0, 40.0])
        self.assertEqual(first_slice["center_source"], "image_center")


if __name__ == "__main__":
    unittest.main()
