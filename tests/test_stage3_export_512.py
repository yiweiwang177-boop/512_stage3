import inspect
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

import stage3_export_512 as export512
import stage3_main_512
import stage3_onh3d_contract
from stage3_onh3d_contract import (
    BMOPlane,
    ILMSurfaceModel,
    ONH3DCase,
    ONH3DQCMeta,
    SectorReference,
    TransformInfo,
)
from stage3_onh3d_metrics import (
    ONH3DLocalFrame,
    ONH3DLocalMetric,
    ONH3DMetricConfig,
    ONH3DMetricResult,
    ONH3DRingSample,
    ONH3DSectorMetricSummary,
    build_ring_geometry,
)
from stage3_onh3d_report_adapter import adapt_onh3d_metrics_to_stage3_tables


def build_case():
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
    return ONH3DCase(
        case_id="CASE-512",
        patient_id="P512",
        laterality="R",
        axial_length=24.2,
        algorithm_version="v2",
        geometry_model="true_3d_onh",
        bmo_ring_3d=ring,
        bmo_plane=BMOPlane(
            origin_3d=np.array([0.0, 0.0, 0.0], dtype=float),
            normal_3d=np.array([0.0, 0.0, 1.0], dtype=float),
            x_axis_3d=np.array([1.0, 0.0, 0.0], dtype=float),
            y_axis_3d=np.array([0.0, 1.0, 0.0], dtype=float),
        ),
        sector_reference=SectorReference(
            angle_zero_axis_3d=np.array([1.0, 0.0, 0.0], dtype=float),
            clockwise_positive=True,
        ),
        transform_info=TransformInfo(world_unit="mm"),
        ilm_surface=ILMSurfaceModel(
            vertices_3d=np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=float),
            faces=np.array([[0, 1, 2]], dtype=int),
        ),
        qc_meta=ONH3DQCMeta(
            bmo_ring_closed=True,
            bmo_ring_sampling_count=8,
            ilm_surface_vertex_count=3,
            ilm_surface_face_count=1,
            bmo_plane_fit_rms_mm=0.0,
            notes=[],
        ),
    )


def build_metric_result(case):
    frame = ONH3DLocalFrame(
        origin_3d=np.array([0.0, 0.0, 0.0], dtype=float),
        tangent_3d=np.array([1.0, 0.0, 0.0], dtype=float),
        reference_normal_3d=np.array([0.0, 0.0, 1.0], dtype=float),
        plane_x_axis_3d=np.array([1.0, 0.0, 0.0], dtype=float),
        plane_y_axis_3d=np.array([0.0, 1.0, 0.0], dtype=float),
        plane_normal_3d=np.array([0.0, -1.0, 0.0], dtype=float),
    )
    sample = ONH3DRingSample(
        sample_idx=0,
        arc_length_s_mm=0.0,
        delta_s_mm=0.5,
        bmo_point_3d=np.array([0.0, 0.0, 0.0], dtype=float),
        bmo_tangent_3d=np.array([1.0, 0.0, 0.0], dtype=float),
        theta_ref_deg=25.0,
        sector_8="NS (Nasal Superior / 姒ц绗傛笟?)",
        sector_4="Nasal (Nasal / 姒ц鏅?",
        sector_2="Superior Half (Superior Half / 娑撳﹤宕愰柈?)",
        local_frame=frame,
    )
    detail = [
        ONH3DLocalMetric(
            sample_idx=0,
            is_valid=True,
            invalid_reason=None,
            bmo_point_3d=np.array([0.0, 0.0, 0.0], dtype=float),
            ilm_point_3d=np.array([0.0, 1.0, 0.0], dtype=float),
            connection_vec_3d=np.array([0.0, 1.0, 0.0], dtype=float),
            mrw_um=999.0,
            phi_deg=89.0,
            delta_s_mm=0.5,
            mra_contrib_mm2=1.234,
            theta_ref_deg=25.0,
            sector_8=sample.sector_8,
            sector_4=sample.sector_4,
            sector_2=sample.sector_2,
        )
    ]
    return ONH3DMetricResult(
        config=ONH3DMetricConfig(ring_sample_count=128),
        ring_geometry=build_ring_geometry(case.bmo_ring_3d),
        ring_samples=[sample],
        detail=detail,
        sector_summary_8={sample.sector_8: ONH3DSectorMetricSummary(8, sample.sector_8, 999.0, 999.0, 1.234, 1)},
        sector_summary_4={sample.sector_4: ONH3DSectorMetricSummary(4, sample.sector_4, 999.0, 999.0, 1.234, 1)},
        sector_summary_2={sample.sector_2: ONH3DSectorMetricSummary(2, sample.sector_2, 999.0, 999.0, 1.234, 1)},
        MRW_global_mean_um=123.0,
        MRW_global_min_um=111.0,
        MRW_global_low10_mean_um=115.0,
        MRA_global_sum_mm2=7.5,
        valid_sample_count=1,
        total_sample_count=128,
        meta={},
        qc={},
    )


class Stage3Export512Tests(unittest.TestCase):
    def setUp(self):
        workspace_tmp = Path(__file__).resolve().parents[1] / "tests_tmp"
        workspace_tmp.mkdir(exist_ok=True)
        self.root = workspace_tmp / f"export512_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_contract_no_longer_depends_on_stage3_reporting(self):
        source = inspect.getsource(stage3_onh3d_contract)
        self.assertIn("from stage3_sector_reference import", source)
        self.assertNotIn("from stage3_reporting import", source)

    def test_build_master_table_512_uses_metric_result_global_summary(self):
        case = build_case()
        result = build_metric_result(case)
        adapted = adapt_onh3d_metrics_to_stage3_tables(case, result)
        mrw_df = adapted["mrw_df"].copy()
        mrw_df.loc[0, "mrw_len_um"] = 1.0
        mra_df = adapted["mra_df"].copy()
        mra_df.loc[0, "local_area_mm2"] = 9999.0

        master_df = export512.build_master_table_512(
            case,
            result,
            mrw_df,
            mra_df,
            adapted["sector_df"],
            adapted["report_meta"],
        )
        row = master_df.iloc[0]
        self.assertEqual(row["MRW_global_mean_um"], 123.0)
        self.assertEqual(row["MRW_global_min_um"], 111.0)
        self.assertEqual(row["MRW_global_low10_mean_um"], 115.0)
        self.assertEqual(row["MRA_global_sum_mm2"], 7.5)

    def test_build_run_summary_df_512_has_no_lcd_lcci_fields(self):
        case = build_case()
        result = build_metric_result(case)
        adapted = adapt_onh3d_metrics_to_stage3_tables(case, result)
        run_summary_df = export512.build_run_summary_df_512(case, result, adapted["report_meta"])
        cols = set(run_summary_df.columns)
        self.assertNotIn("mean_lcd_area_mm", cols)
        self.assertNotIn("mean_lcci_area_mm", cols)
        self.assertNotIn("LCD_direct_global_mean", cols)
        self.assertNotIn("LCCI_direct_global_mean", cols)

    def test_export_results_excel_512_omits_lcd_lcci_sheet(self):
        case = build_case()
        result = build_metric_result(case)
        adapted = adapt_onh3d_metrics_to_stage3_tables(case, result)
        master_df = export512.build_master_table_512(
            case,
            result,
            adapted["mrw_df"],
            adapted["mra_df"],
            adapted["sector_df"],
            adapted["report_meta"],
        )
        run_summary_df = export512.build_run_summary_df_512(case, result, adapted["report_meta"])
        output_path = self.root / "out.xlsx"
        export512.export_results_excel_512(
            str(output_path),
            master_df,
            run_summary_df,
            adapted["mrw_df"],
            adapted["mra_df"],
            adapted["sector_df"],
        )
        with pd.ExcelFile(output_path) as xls:
            self.assertEqual(
                xls.sheet_names,
                ["Master_Table", "Run_Summary", "MRW_detail", "MRA_detail", "Sector_Summary"],
            )

    def test_stage3_main_512_no_longer_imports_old_reporting_api(self):
        source = inspect.getsource(stage3_main_512)
        self.assertIn("from stage3_export_512 import", source)
        self.assertNotIn("from stage3_reporting import", source)

    def test_sector_summary_uses_value_schema_without_mean_value_bridge(self):
        case = build_case()
        result = build_metric_result(case)
        adapted = adapt_onh3d_metrics_to_stage3_tables(case, result)
        self.assertIn("value", adapted["sector_df"].columns)
        self.assertIn("aggregation_method", adapted["sector_df"].columns)
        self.assertNotIn("mean_value", adapted["sector_df"].columns)
        master_df = export512.build_master_table_512(
            case,
            result,
            adapted["mrw_df"],
            adapted["mra_df"],
            adapted["sector_df"],
            adapted["report_meta"],
        )
        self.assertIn("MRW_global_mean_um", master_df.columns)


if __name__ == "__main__":
    unittest.main()
