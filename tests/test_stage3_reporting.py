import shutil
import unittest
import uuid
from pathlib import Path

import pandas as pd

import stage3_reporting as reporting


class Stage3ReportingSmokeTests(unittest.TestCase):
    def setUp(self):
        workspace_tmp = Path(__file__).resolve().parents[1] / "tests_tmp"
        workspace_tmp.mkdir(exist_ok=True)
        self.root = workspace_tmp / f"reporting_smoke_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_expected_master_sector_schema_has_fixed_columns(self):
        schema = reporting.get_expected_master_sector_schema()
        self.assertIn("ordered_columns", schema)
        self.assertIn("labels_by_level", schema)
        self.assertTrue(schema["ordered_columns"])

    def test_master_summary_and_export_smoke(self):
        sector_schema = reporting.get_expected_master_sector_schema()
        sector_rows = [
            {
                "source_table": "MRW_detail",
                "level": "sector_8",
                "sector_name": sector_schema["labels_by_level"]["sector_8"][0],
                "metric_name": "MRW_um",
                "mean_value": 100.0,
                "count": 2,
            },
            {
                "source_table": "MRA_detail",
                "level": "sector_8",
                "sector_name": sector_schema["labels_by_level"]["sector_8"][0],
                "metric_name": "MRA_local_area_mm2",
                "mean_value": 1.5,
                "count": 2,
            },
        ]
        master_df = reporting.build_master_table(
            case_id="CASE-1",
            patient_id="P001",
            laterality="R",
            axial_length=24.0,
            baseline_row={"Patient_Name": "Alice"},
            stage2_schema_version="v1",
            z_stabilization_status="stable",
            self_check_df=pd.DataFrame([{"Check": "a", "Status": "PASS", "Detail": "ok"}]),
            final_cloud={"SLICE_META": {}},
            mrw_df=pd.DataFrame({"mrw_len_um": [100.0, 110.0]}),
            global_mra_mm2=9.0,
            lcd_lcci_df=pd.DataFrame(columns=["status"]),
            sector_df=pd.DataFrame(sector_rows),
        )
        self.assertEqual(master_df.iloc[0]["MRW_global_mean"], 105.0)

        summary_df = reporting.build_run_summary_df(
            master_df,
            pd.DataFrame(columns=["status"]),
            workbook_path=str(self.root / "out.xlsx"),
            qc_slice_dir=str(self.root / "QC_Slices"),
            timestamp="2026-04-01T00:00:00",
        )
        self.assertEqual(summary_df.iloc[0]["global_mra_mm2"], 9.0)

        workbook_path = self.root / "report.xlsx"
        reporting.export_results_excel(
            str(workbook_path),
            master_df,
            summary_df,
            pd.DataFrame([{"Check": "a", "Status": "PASS", "Detail": "ok"}]),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(sector_rows),
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

    def test_export_sheet_order_unaffected_by_value_based_sector_summary(self):
        master_df = pd.DataFrame([{"case_id": "C1"}])
        run_summary_df = pd.DataFrame([{"case_id": "C1"}])
        workbook_path = self.root / "report_value_sector.xlsx"
        reporting.export_results_excel(
            str(workbook_path),
            master_df,
            run_summary_df,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(
                [
                    {
                        "source_table": "MRW_detail",
                        "level": "sector_8",
                        "sector_name": "S8",
                        "metric_name": "MRW_um",
                        "value": 1.0,
                        "aggregation_method": "mean",
                        "count": 1,
                    }
                ]
            ),
        )
        with pd.ExcelFile(workbook_path) as xls:
            self.assertEqual(xls.sheet_names[-1], "Sector_Summary")


if __name__ == "__main__":
    unittest.main()
