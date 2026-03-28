import importlib.util
import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock

import pandas as pd

from stage3_input_adapter import (
    load_patient_baseline_row,
    load_stage2_case,
    validate_stage3_input_contract,
)
from stage3_shared import build_stage3_shared_structure


def build_stage2_case_payload(n_slices=12):
    slices = []
    for scan_index in range(1, n_slices + 1):
        full_ilm = [[float(x), 20.0 + 0.1 * scan_index] for x in range(0, 10)]
        slices.append(
            {
                "scan_index": scan_index,
                "slice_stem": f"slice_{scan_index:02d}",
                "image_width": 10,
                "image_height": 100,
                "full_ilm_px": full_ilm,
                "bmo_px": [[3.0, 40.0], [6.0, 40.0]],
                "cutoff_px": [[2.0, 35.0], [7.0, 35.0]],
                "rnfl_effective_lower_px": [[2.0, 60.0], [4.0, 59.0], [7.0, 60.0]],
                "review_status": "approved",
                "source_flags": {"stage2_final": True},
            }
        )
    return {
        "stage2_schema_version": "v1",
        "case_id": "CASE-001",
        "patient_id": "P001",
        "laterality": "R",
        "slices": slices,
    }


def load_stage3_main_module():
    root = Path(__file__).resolve().parents[1]
    target = root / "最终.py"
    spec = importlib.util.spec_from_file_location("stage3_main_module", str(target))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Stage3InputAdapterTests(unittest.TestCase):
    def setUp(self):
        workspace_tmp = Path(__file__).resolve().parents[1] / "tests_tmp"
        workspace_tmp.mkdir(exist_ok=True)
        self.root = workspace_tmp / f"input_adapter_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.stage2_json = self.root / "case.json"
        self.base_table = self.root / "baseline.xlsx"

        payload = build_stage2_case_payload()
        with open(self.stage2_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

        baseline_df = pd.DataFrame(
            [
                {
                    "Patient_ID": "P001",
                    "Laterality": "R",
                    "Axial_Length": 24.0,
                    "Diagnosis": "TestDx",
                    "Stage": "S1",
                }
            ]
        )
        baseline_df.to_excel(self.base_table, index=False)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_load_stage2_case_success(self):
        loaded = load_stage2_case(str(self.stage2_json), expected_case_id="CASE-001")
        self.assertEqual(loaded["stage2_schema_version"], "v1")
        self.assertEqual(loaded["case_id"], "CASE-001")
        self.assertEqual(loaded["laterality"], "R")
        self.assertEqual(len(loaded["slices"]), 12)

    def test_validate_contract_requires_12_slices(self):
        payload = build_stage2_case_payload(n_slices=11)
        with open(self.stage2_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

        loaded = load_stage2_case(str(self.stage2_json), expected_case_id="CASE-001")
        baseline_row = load_patient_baseline_row(str(self.base_table), "P001", "R")
        records = validate_stage3_input_contract(loaded, baseline_row)
        status_map = {item["Check"]: item["Status"] for item in records}

        self.assertEqual(status_map["stage2:schema_version_present"], "PASS")
        self.assertEqual(status_map["stage2:schema_version_supported"], "PASS")
        self.assertEqual(status_map["stage2:slice_count"], "FAIL")
        self.assertEqual(status_map["stage2:scan_indices_complete"], "FAIL")

    def test_validate_contract_requires_required_fields(self):
        payload = build_stage2_case_payload()
        del payload["slices"][0]["full_ilm_px"]
        with open(self.stage2_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

        with self.assertRaisesRegex(ValueError, "missing slice field: full_ilm_px"):
            load_stage2_case(str(self.stage2_json), expected_case_id="CASE-001")

    def test_stage2_schema_version_required(self):
        payload = build_stage2_case_payload()
        del payload["stage2_schema_version"]
        with open(self.stage2_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

        with self.assertRaisesRegex(ValueError, "missing case field: stage2_schema_version"):
            load_stage2_case(str(self.stage2_json), expected_case_id="CASE-001")

    def test_stage2_schema_version_rejects_unknown_value(self):
        payload = build_stage2_case_payload()
        payload["stage2_schema_version"] = "v2"
        with open(self.stage2_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

        with self.assertRaisesRegex(ValueError, "unsupported stage2_schema_version"):
            load_stage2_case(str(self.stage2_json), expected_case_id="CASE-001")

    def test_scan_index_and_angle_preserved(self):
        loaded = load_stage2_case(str(self.stage2_json), expected_case_id="CASE-001")
        baseline_row = load_patient_baseline_row(str(self.base_table), "P001", "R")
        shared_case = build_stage3_shared_structure(loaded, baseline_row)

        first_slice = shared_case.slice_meta_list[0]
        self.assertEqual(first_slice.scan_index, 1)
        self.assertEqual(first_slice.angle_deg, 0.0)
        self.assertEqual(first_slice.x_center, 5.0)
        self.assertEqual(first_slice.y_center, 50.0)

        last_slice = shared_case.slice_meta_list[-1]
        self.assertEqual(last_slice.scan_index, 12)
        self.assertEqual(last_slice.angle_deg, 165.0)

    def test_build_shared_structure_contains_case_slice_compute_levels(self):
        loaded = load_stage2_case(str(self.stage2_json), expected_case_id="CASE-001")
        baseline_row = load_patient_baseline_row(str(self.base_table), "P001", "R")
        shared_case = build_stage3_shared_structure(loaded, baseline_row)

        self.assertEqual(shared_case.case_meta.case_id, "CASE-001")
        self.assertEqual(len(shared_case.slice_meta_list), 12)
        self.assertIn("BMO_META", shared_case.compute_meta)
        self.assertIn("ILM_FULL_META", shared_case.compute_meta)
        self.assertIn("CUT_POINTS_META", shared_case.compute_meta)
        self.assertIn("RNFL_EFFECTIVE_SEG_META", shared_case.compute_meta)
        self.assertIn("SLICE_META", shared_case.compute_meta)

    def test_stage2_mode_does_not_require_auto_discovery(self):
        module = load_stage3_main_module()
        output_dir = self.root / "out"
        captured = {}

        def capture_align(cloud):
            captured["cloud"] = cloud
            return {
                "laterality": "R",
                "axial_length": 24.0,
                "z_stabilization_status": cloud.get("z_stabilization_status"),
                "BMO_META": [],
                "ALI_META": [],
                "ALCS_META": [],
                "SLICE_META": cloud.get("SLICE_META", {}),
                "BMO": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
            }

        with mock.patch.object(
            module,
            "auto_discover_paths",
            side_effect=AssertionError("auto_discover_paths should not be called in stage2 mode"),
        ), mock.patch.object(
            module,
            "load_legacy_labelme_case",
            side_effect=AssertionError("load_legacy_labelme_case should not be called in stage2 mode"),
        ), mock.patch.object(module, "align_to_bmo_bfp") as mock_align, mock.patch.object(
            module, "prepare_mrw_dataframe", return_value=pd.DataFrame()
        ), mock.patch.object(
            module, "calculate_gardiner_mra", return_value=([], 0.0)
        ), mock.patch.object(
            module, "prepare_mra_dataframe", return_value=pd.DataFrame()
        ), mock.patch.object(
            module, "compute_traditional_lcd_lcci_all_slices", return_value=(pd.DataFrame(), {})
        ), mock.patch.object(
            module, "prepare_lcd_lcci_dataframe", return_value=pd.DataFrame()
        ), mock.patch.object(
            module, "build_sector_summary_from_tables", return_value=pd.DataFrame()
        ), mock.patch.object(
            module, "save_slice_qc_figures", return_value=None
        ), mock.patch.object(
            module, "export_results_excel", return_value=None
        ):
            mock_align.side_effect = capture_align

            result = module.main(
                [
                    "--input-mode",
                    "stage2",
                    "--stage2-json",
                    str(self.stage2_json),
                    "--base-table",
                    str(self.base_table),
                    "--case-id",
                    "CASE-001",
                    "--patient-id",
                    "P001",
                    "--laterality",
                    "R",
                    "--output-dir",
                    str(output_dir),
                ]
            )

        self.assertEqual(result["status"], "OK")
        self.assertIn("cloud", captured)
        first_slice = captured["cloud"]["SLICE_META"][1]
        self.assertEqual(first_slice["x_center"], 5.0)
        self.assertEqual(first_slice["y_center"], 50.0)
        self.assertEqual(first_slice["rotation_center_px"], [5.0, 50.0])
        self.assertEqual(first_slice["center_source"], "image_center")
        self.assertEqual(captured["cloud"]["z_stabilization_status"], "inactive_stage2_no_z_correction")

    def test_stage2_mode_does_not_require_center_ilm(self):
        module = load_stage3_main_module()
        output_dir = self.root / "out_no_center_ilm"
        payload = build_stage2_case_payload()
        for item in payload["slices"]:
            item.pop("source_flags", None)
        with open(self.stage2_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

        with mock.patch.object(module, "align_to_bmo_bfp") as mock_align, mock.patch.object(
            module, "prepare_mrw_dataframe", return_value=pd.DataFrame()
        ), mock.patch.object(
            module, "calculate_gardiner_mra", return_value=([], 0.0)
        ), mock.patch.object(
            module, "prepare_mra_dataframe", return_value=pd.DataFrame()
        ), mock.patch.object(
            module, "compute_traditional_lcd_lcci_all_slices", return_value=(pd.DataFrame(), {})
        ), mock.patch.object(
            module, "prepare_lcd_lcci_dataframe", return_value=pd.DataFrame()
        ), mock.patch.object(
            module, "build_sector_summary_from_tables", return_value=pd.DataFrame()
        ), mock.patch.object(
            module, "save_slice_qc_figures", return_value=None
        ), mock.patch.object(
            module, "export_results_excel", return_value=None
        ):
            mock_align.return_value = {
                "laterality": "R",
                "axial_length": 24.0,
                "z_stabilization_status": "inactive_stage2_no_z_correction",
                "BMO_META": [],
                "ALI_META": [],
                "ALCS_META": [],
                "SLICE_META": {},
                "BMO": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
            }
            result = module.main(
                [
                    "--input-mode",
                    "stage2",
                    "--stage2-json",
                    str(self.stage2_json),
                    "--base-table",
                    str(self.base_table),
                    "--case-id",
                    "CASE-001",
                    "--patient-id",
                    "P001",
                    "--laterality",
                    "R",
                    "--output-dir",
                    str(output_dir),
                ]
            )

        self.assertEqual(result["status"], "OK")


if __name__ == "__main__":
    unittest.main()
