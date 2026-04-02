import json
import shutil
import unittest
import uuid
from pathlib import Path

import pandas as pd

from stage3_export_512 import (
    build_master_table_512,
    build_run_summary_df_512,
    export_results_excel_512,
)
from stage3_main_512 import load_onh3d_case, main_512
from stage3_onh3d_contract import ONH3DCase, validate_onh3d_case
from stage3_onh3d_metrics import ONH3DMetricResult, compute_onh3d_metrics
from stage3_onh3d_report_adapter import adapt_onh3d_metrics_to_stage3_tables


def build_minimal_onh3d_payload():
    return {
        "case_id": "CASE-SMOKE",
        "patient_id": "P-SMOKE",
        "laterality": "R",
        "axial_length": 24.1,
        "algorithm_version": "smoke-v1",
        "geometry_model": "true_3d_onh",
        "bmo_ring_3d": {
            "vertices_3d": [
                [1.0, 0.0, 0.0],
                [0.7071, 0.7071, 0.0],
                [0.0, 1.0, 0.0],
                [-0.7071, 0.7071, 0.0],
                [-1.0, 0.0, 0.0],
                [-0.7071, -0.7071, 0.0],
                [0.0, -1.0, 0.0],
                [0.7071, -0.7071, 0.0],
            ],
            "world_frame": "patient_mm",
            "notes": ["canonical_ring_structure"],
        },
        "ilm_surface": {
            "vertices_3d": [
                [-2.0, -2.0, 1.0],
                [2.0, -2.0, 1.0],
                [2.0, 2.0, 1.0],
                [-2.0, 2.0, 1.0],
            ],
            "faces": [
                [0, 1, 2],
                [0, 2, 3],
            ],
            "vertex_normals_3d": [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            "surface_bounds_3d": [
                [-2.0, -2.0, 1.0],
                [2.0, 2.0, 1.0],
            ],
        },
        "bmo_plane": {
            "origin_3d": [0.0, 0.0, 0.0],
            "normal_3d": [0.0, 0.0, 1.0],
            "x_axis_3d": [1.0, 0.0, 0.0],
            "y_axis_3d": [0.0, 1.0, 0.0],
        },
        "sector_reference": {
            "angle_zero_axis_3d": [1.0, 0.0, 0.0],
            "clockwise_positive": True,
        },
        "transform_info": {
            "world_unit": "mm",
            "voxel_spacing_mm_xyz": [0.01, 0.01, 0.01],
            "voxel_to_world": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "qc_meta": {
            "bmo_ring_closed": True,
            "bmo_ring_sampling_count": 8,
            "ilm_surface_vertex_count": 4,
            "ilm_surface_face_count": 2,
            "bmo_plane_fit_rms_mm": 0.0,
            "notes": ["stage12_canonical_artifact"],
        },
        "source_meta": {
            "upstream_case_format": "canonical_onh3d_case_json",
        },
    }


class Stage3Main512SmokeTests(unittest.TestCase):
    def setUp(self):
        workspace_tmp = Path(__file__).resolve().parents[1] / "tests_tmp"
        workspace_tmp.mkdir(exist_ok=True)
        self.root = workspace_tmp / f"main512_smoke_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.case_path = self.root / "case_onh3d.json"
        with self.case_path.open("w", encoding="utf-8") as handle:
            json.dump(build_minimal_onh3d_payload(), handle, ensure_ascii=False, indent=2)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_end_to_end_smoke_without_old_entrypoints(self):
        case = load_onh3d_case(str(self.case_path))
        self.assertIsInstance(case, ONH3DCase)
        validate_onh3d_case(case)
        self.assertEqual(case.source_meta["loader_contract"], "onh3d_case_json_v1")
        self.assertEqual(case.source_meta["loader_bmo_ring_geometry_source"], "bmo_ring_3d.vertices_3d")
        self.assertIn("world_frame", case.source_meta["loader_bmo_ring_metadata"])
        self.assertNotIn("loader_defaults_applied", case.source_meta)

        metrics_result = compute_onh3d_metrics(case)
        self.assertIsInstance(metrics_result, ONH3DMetricResult)
        self.assertEqual(metrics_result.total_sample_count, 128)
        self.assertGreater(metrics_result.valid_sample_count, 0)
        self.assertFalse(pd.isna(metrics_result.MRW_global_mean_um))
        self.assertFalse(pd.isna(metrics_result.MRW_global_min_um))
        self.assertFalse(pd.isna(metrics_result.MRW_global_low10_mean_um))
        self.assertGreater(metrics_result.MRA_global_sum_mm2, 0.0)

        adapted = adapt_onh3d_metrics_to_stage3_tables(case, metrics_result)
        self.assertIn("mrw_df", adapted)
        self.assertIn("mra_df", adapted)
        self.assertIn("sector_df", adapted)
        self.assertIn("report_meta", adapted)

        master_df = build_master_table_512(
            case,
            metrics_result,
            adapted["mrw_df"],
            adapted["mra_df"],
            adapted["sector_df"],
            adapted["report_meta"],
        )
        run_summary_df = build_run_summary_df_512(case, metrics_result, adapted["report_meta"])
        self.assertIsInstance(master_df, pd.DataFrame)
        self.assertIsInstance(run_summary_df, pd.DataFrame)
        self.assertGreater(len(adapted["mrw_df"][adapted["mrw_df"]["connection_valid"] == True]), 0)

        output_path = self.root / "smoke_export.xlsx"
        export_results_excel_512(
            str(output_path),
            master_df,
            run_summary_df,
            adapted["mrw_df"],
            adapted["mra_df"],
            adapted["sector_df"],
        )
        self.assertTrue(output_path.is_file())
        with pd.ExcelFile(output_path) as xls:
            self.assertEqual(
                xls.sheet_names,
                ["Master_Table", "Run_Summary", "MRW_detail", "MRA_detail", "Sector_Summary"],
            )

    def test_main_512_writes_excel_when_output_dir_is_provided(self):
        output_dir = self.root / "outputs"
        result = main_512(
            [
                "--onh3d-case",
                str(self.case_path),
                "--output-dir",
                str(output_dir),
                "--overwrite",
            ]
        )
        self.assertIn("master_df", result)
        self.assertIn("run_summary_df", result)
        self.assertIn("metrics_result", result)
        self.assertGreater(result["metrics_result"].valid_sample_count, 0)
        workbook_path = output_dir / "CASE-SMOKE_stage3_512.xlsx"
        self.assertTrue(workbook_path.is_file())


if __name__ == "__main__":
    unittest.main()
