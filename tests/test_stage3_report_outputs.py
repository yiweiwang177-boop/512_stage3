import importlib.util
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


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


class Stage3ReportOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_stage3_main_module()

    def setUp(self):
        workspace_tmp = Path(__file__).resolve().parents[1] / "tests_tmp"
        workspace_tmp.mkdir(exist_ok=True)
        self.root = workspace_tmp / f"report_outputs_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def build_baseline_row(self):
        return {
            "Patient_ID": "P001",
            "Laterality": "R",
            "Axial_Length": 24.5,
            "Patient_Name": "Alice",
            "Sex": "F",
            "Age": 61,
            "Diagnosis": "POAG",
            "Stage": "S2",
            "CCT": 525,
        }

    def build_sector_df(self):
        return pd.DataFrame(
            [
                {
                    "source_table": "MRW_detail",
                    "level": "sector_8",
                    "sector_name": "Alpha (One)/Two",
                    "metric_name": "MRW_um",
                    "mean_value": 11.0,
                    "count": 2,
                },
                {
                    "source_table": "MRW_detail",
                    "level": "sector_4",
                    "sector_name": "Beta / Three",
                    "metric_name": "MRW_um",
                    "mean_value": 12.0,
                    "count": 2,
                },
                {
                    "source_table": "MRW_detail",
                    "level": "sector_2",
                    "sector_name": "Gamma (Four)",
                    "metric_name": "MRW_um",
                    "mean_value": 13.0,
                    "count": 2,
                },
                {
                    "source_table": "MRA_detail",
                    "level": "sector_8",
                    "sector_name": "Delta / Five",
                    "metric_name": "MRA_local_area_mm2",
                    "mean_value": 1.1,
                    "count": 2,
                },
                {
                    "source_table": "MRA_detail",
                    "level": "sector_4",
                    "sector_name": "Epsilon (Six)",
                    "metric_name": "MRA_local_area_mm2",
                    "mean_value": 1.2,
                    "count": 2,
                },
                {
                    "source_table": "MRA_detail",
                    "level": "sector_2",
                    "sector_name": "Zeta / Seven",
                    "metric_name": "MRA_local_area_mm2",
                    "mean_value": 1.3,
                    "count": 2,
                },
                {
                    "source_table": "LCD_LCCI_detail",
                    "level": "sector_8",
                    "sector_name": "Ignored / Eight",
                    "metric_name": "LCD_area_mm",
                    "mean_value": 2.2,
                    "count": 2,
                },
            ]
        )

    def build_master_inputs(self):
        baseline_row = self.build_baseline_row()
        mrw_df = pd.DataFrame({"mrw_len_um": [100.0, 200.0]})
        lcd_lcci_df = pd.DataFrame(
            [
                {
                    "status": "PASS",
                    "lcd_area_mm": 1.0,
                    "lcci_area_mm": 2.0,
                    "lcd_direct_mm": 10.0,
                    "lcci_direct_mm": 20.0,
                    "alcci_area_percent": 30.0,
                    "alcci_direct_percent": 40.0,
                },
                {
                    "status": "FAIL",
                    "lcd_area_mm": 100.0,
                    "lcci_area_mm": 200.0,
                    "lcd_direct_mm": 1000.0,
                    "lcci_direct_mm": 2000.0,
                    "alcci_area_percent": 300.0,
                    "alcci_direct_percent": 400.0,
                },
                {
                    "status": "PASS",
                    "lcd_area_mm": 3.0,
                    "lcci_area_mm": 4.0,
                    "lcd_direct_mm": 30.0,
                    "lcci_direct_mm": 40.0,
                    "alcci_area_percent": 50.0,
                    "alcci_direct_percent": 60.0,
                },
            ]
        )
        sector_df = self.build_sector_df()
        self_check_df = pd.DataFrame(
            [
                {"Check": "a", "Status": "PASS", "Detail": "ok"},
                {"Check": "b", "Status": "FAIL", "Detail": "problem"},
            ]
        )
        final_cloud = {
            "SLICE_META": {
                1: {"review_status": "approved"},
                2: {"review_status": "approved"},
            }
        }
        return baseline_row, mrw_df, lcd_lcci_df, sector_df, self_check_df, final_cloud

    def test_master_table_structure_and_sector_order(self):
        baseline_row, mrw_df, lcd_lcci_df, sector_df, self_check_df, final_cloud = self.build_master_inputs()

        master_df = self.module.build_master_table(
            case_id="CASE-001",
            patient_id="P001",
            laterality="R",
            axial_length=24.5,
            baseline_row=baseline_row,
            stage2_schema_version="v1",
            z_stabilization_status="stable",
            self_check_df=self_check_df,
            final_cloud=final_cloud,
            mrw_df=mrw_df,
            global_mra_mm2=9.5,
            lcd_lcci_df=lcd_lcci_df,
            sector_df=sector_df,
        )

        columns = list(master_df.columns)
        self.assertEqual(columns[:4], ["case_id", "patient_id", "laterality", "axial_length"])
        self.assertEqual(
            columns[4:10],
            ["Patient_Name", "Sex", "Age", "Diagnosis", "Stage", "CCT"],
        )
        self.assertNotIn("Patient_ID", columns)
        self.assertNotIn("Laterality", columns)
        self.assertNotIn("Axial_Length", columns)

        row = master_df.iloc[0]
        self.assertEqual(row["MRW_global_mean"], 150.0)
        self.assertEqual(row["MRA_global_mm2"], 9.5)
        self.assertEqual(row["LCD_global_mean"], 2.0)
        self.assertEqual(row["LCCI_global_mean"], 3.0)

        sector_columns = [col for col in columns if col.startswith("MRW_sector_") or col.startswith("MRA_sector_")]
        self.assertEqual(
            sector_columns,
            [
                "MRW_sector_8_alpha_one_two_um",
                "MRW_sector_4_beta_three_um",
                "MRW_sector_2_gamma_four_um",
                "MRA_sector_8_delta_five_mm2",
                "MRA_sector_4_epsilon_six_mm2",
                "MRA_sector_2_zeta_seven_mm2",
            ],
        )
        self.assertFalse(any(col.startswith("LCD_sector_") or col.startswith("LCCI_sector_") for col in columns))

    def test_master_table_uses_pass_slice_area_only_for_lcd_lcci_globals(self):
        baseline_row, mrw_df, lcd_lcci_df, sector_df, self_check_df, final_cloud = self.build_master_inputs()
        master_df = self.module.build_master_table(
            case_id="CASE-001",
            patient_id="P001",
            laterality="R",
            axial_length=24.5,
            baseline_row=baseline_row,
            stage2_schema_version="v1",
            z_stabilization_status="stable",
            self_check_df=self_check_df,
            final_cloud=final_cloud,
            mrw_df=mrw_df,
            global_mra_mm2=9.5,
            lcd_lcci_df=lcd_lcci_df,
            sector_df=sector_df,
        )

        row = master_df.iloc[0]
        self.assertEqual(row["LCD_global_mean"], 2.0)
        self.assertEqual(row["LCCI_global_mean"], 3.0)
        self.assertEqual(row["lcd_lcci_pass_slices"], 2)
        self.assertEqual(row["lcd_lcci_total_slices"], 3)
        self.assertTrue(row["lcd_lcci_pass_available"])
        self.assertNotIn("LCD_direct_global_mean", master_df.columns)
        self.assertNotIn("LCCI_direct_global_mean", master_df.columns)

    def test_master_table_no_pass_slices_emits_nan_and_status_flag(self):
        baseline_row, mrw_df, _, sector_df, self_check_df, final_cloud = self.build_master_inputs()
        lcd_lcci_df = pd.DataFrame(
            [
                {"status": "FAIL", "lcd_area_mm": 1.0, "lcci_area_mm": 2.0},
                {"status": "FAIL", "lcd_area_mm": 3.0, "lcci_area_mm": 4.0},
            ]
        )

        master_df = self.module.build_master_table(
            case_id="CASE-001",
            patient_id="P001",
            laterality="R",
            axial_length=24.5,
            baseline_row=baseline_row,
            stage2_schema_version="v1",
            z_stabilization_status="stable",
            self_check_df=self_check_df,
            final_cloud=final_cloud,
            mrw_df=mrw_df,
            global_mra_mm2=9.5,
            lcd_lcci_df=lcd_lcci_df,
            sector_df=sector_df,
        )

        row = master_df.iloc[0]
        self.assertTrue(pd.isna(row["LCD_global_mean"]))
        self.assertTrue(pd.isna(row["LCCI_global_mean"]))
        self.assertEqual(row["lcd_lcci_pass_slices"], 0)
        self.assertFalse(row["lcd_lcci_pass_available"])

    def test_prepare_mra_dataframe_includes_phi_signed_deg(self):
        gardiner_local_list = [
            {
                "slice_id": 1,
                "side": "L",
                "scan_angle_deg": 0.0,
                "anatomical_angle_deg": 15.0,
                "r_mm": 1.0,
                "bottom_len_mm": 0.5,
                "phi_signed_deg": -12.5,
                "mra_phi_deg": 12.5,
                "rw_phi_um": 100.0,
                "top_len_mm": 0.4,
                "local_area_mm2": 0.2,
                "bmo_pt": [0.0, 0.0, 0.0],
                "ilm_hit_pt": [1.0, 0.0, 1.0],
            }
        ]
        final_cloud = {
            "BMO_META": [
                {"slice_id": 1, "side": "L", "pixel_x": 10.0, "pixel_y": 20.0},
            ]
        }
        aligned_cloud = {
            "SLICE_META": {
                1: {
                    "scale_X": 1.0,
                    "scale_Z": 1.0,
                    "x_center": 0.0,
                    "delta_z": 0.0,
                    "angle_deg": 0.0,
                }
            },
            "ALIGNMENT": {
                "rotation_matrix": np.eye(3).tolist(),
                "centroid": [0.0, 0.0, 0.0],
            },
            "laterality": "R",
        }

        df = self.module.prepare_mra_dataframe(gardiner_local_list, final_cloud, aligned_cloud)

        self.assertIn("phi_signed_deg", df.columns)
        self.assertAlmostEqual(df.iloc[0]["phi_signed_deg"], -12.5, places=6)

    def test_build_sector_summary_uses_canonical_source_names(self):
        mrw_df = pd.DataFrame(
            [{"sector_8_name": "A", "sector_4_name": "B", "sector_2_name": "C", "mrw_len_um": 10.0}]
        )
        mra_df = pd.DataFrame(
            [{"sector_8_name": "A", "sector_4_name": "B", "sector_2_name": "C", "local_area_mm2": 0.5}]
        )
        lcd_lcci_df = pd.DataFrame(
            [
                {
                    "sector_8_name": "A",
                    "sector_4_name": "B",
                    "sector_2_name": "C",
                    "lcd_area_mm": 1.0,
                    "lcd_direct_mm": 2.0,
                    "lcci_area_mm": 3.0,
                    "lcci_direct_mm": 4.0,
                    "alcci_area_percent": 5.0,
                    "alcci_direct_percent": 6.0,
                }
            ]
        )

        sector_df = self.module.build_sector_summary_from_tables(mrw_df, mra_df, lcd_lcci_df)

        self.assertEqual(
            set(sector_df["source_table"]),
            {"MRW_detail", "MRA_detail", "LCD_LCCI_detail"},
        )

    def test_run_summary_is_narrow_and_derived_from_master(self):
        baseline_row, mrw_df, lcd_lcci_df, sector_df, self_check_df, final_cloud = self.build_master_inputs()
        master_df = self.module.build_master_table(
            case_id="CASE-001",
            patient_id="P001",
            laterality="R",
            axial_length=24.5,
            baseline_row=baseline_row,
            stage2_schema_version="v1",
            z_stabilization_status="stable",
            self_check_df=self_check_df,
            final_cloud=final_cloud,
            mrw_df=mrw_df,
            global_mra_mm2=9.5,
            lcd_lcci_df=lcd_lcci_df,
            sector_df=sector_df,
        )
        summary_df = self.module.build_run_summary_df(
            master_df,
            lcd_lcci_df,
            workbook_path=str(self.root / "out.xlsx"),
            qc_slice_dir=str(self.root / "QC_Slices"),
            timestamp="2026-03-28T12:00:00",
        )

        self.assertEqual(summary_df.iloc[0]["mean_mrw_um"], master_df.iloc[0]["MRW_global_mean"])
        self.assertEqual(summary_df.iloc[0]["global_mra_mm2"], master_df.iloc[0]["MRA_global_mm2"])
        self.assertEqual(summary_df.iloc[0]["mean_lcd_area_mm"], master_df.iloc[0]["LCD_global_mean"])
        self.assertEqual(summary_df.iloc[0]["mean_lcci_area_mm"], master_df.iloc[0]["LCCI_global_mean"])
        self.assertEqual(summary_df.iloc[0]["LCD_direct_global_mean"], 20.0)
        self.assertNotIn("Sex", summary_df.columns)
        self.assertFalse(any(col.startswith("MRW_sector_") or col.startswith("MRA_sector_") for col in summary_df.columns))

    def test_export_results_excel_writes_canonical_and_compatibility_sheets(self):
        baseline_row, mrw_df, lcd_lcci_df, sector_df, self_check_df, final_cloud = self.build_master_inputs()
        master_df = self.module.build_master_table(
            case_id="CASE-001",
            patient_id="P001",
            laterality="R",
            axial_length=24.5,
            baseline_row=baseline_row,
            stage2_schema_version="v1",
            z_stabilization_status="stable",
            self_check_df=self_check_df,
            final_cloud=final_cloud,
            mrw_df=mrw_df,
            global_mra_mm2=9.5,
            lcd_lcci_df=lcd_lcci_df,
            sector_df=sector_df,
        )
        summary_df = self.module.build_run_summary_df(
            master_df,
            lcd_lcci_df,
            workbook_path=str(self.root / "out.xlsx"),
            qc_slice_dir=str(self.root / "QC_Slices"),
            timestamp="2026-03-28T12:00:00",
        )
        mra_df = pd.DataFrame([{"phi_signed_deg": -1.0, "mra_phi_deg": 1.0}])
        workbook_path = self.root / "report.xlsx"

        self.module.export_results_excel(
            str(workbook_path),
            master_df,
            summary_df,
            self_check_df,
            mrw_df,
            mra_df,
            lcd_lcci_df,
            sector_df,
        )

        with pd.ExcelFile(workbook_path) as xls:
            self.assertEqual(
                xls.sheet_names,
                [
                    "Master_Table",
                    "Run_Summary",
                    "Self_Check",
                    "MRW_detail",
                    "MRA_detail",
                    "LCD_LCCI_detail",
                    "Sector_Summary",
                ],
            )


if __name__ == "__main__":
    unittest.main()
