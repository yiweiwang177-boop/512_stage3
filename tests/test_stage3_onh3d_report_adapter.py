import unittest

import numpy as np

from stage3_onh3d_contract import (
    BMOPlane,
    ILMSurfaceModel,
    ONH3DCase,
    ONH3DQCMeta,
    SectorReference,
    TransformInfo,
)
from stage3_onh3d_report_adapter import (
    ONH3D_MRA_DETAIL_COLUMNS,
    ONH3D_MRW_DETAIL_COLUMNS,
    ONH3D_SECTOR_SUMMARY_COLUMNS,
    adapt_onh3d_metrics_to_stage3_tables,
    build_empty_onh3d_mra_detail_df,
    build_empty_onh3d_mrw_detail_df,
    build_onh3d_report_meta,
)
from stage3_onh3d_metrics import (
    ONH3DLocalFrame,
    ONH3DLocalMetric,
    ONH3DMetricConfig,
    ONH3DMetricResult,
    ONH3DRingGeometry,
    ONH3DRingSample,
    ONH3DSectorMetricSummary,
    build_ring_geometry,
)


def build_case():
    return ONH3DCase(
        case_id="CASE-512",
        patient_id="P512",
        laterality="R",
        axial_length=24.3,
        algorithm_version="v0",
        geometry_model="true_3d_onh",
        bmo_ring_3d=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]], dtype=float),
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
            bmo_ring_sampling_count=3,
            ilm_surface_vertex_count=3,
            ilm_surface_face_count=1,
            bmo_plane_fit_rms_mm=0.0,
            notes=[],
        ),
    )


class Stage3ONH3DReportAdapterTests(unittest.TestCase):
    def test_detail_column_sets_include_required_and_compat_columns(self):
        for col in ["sample_index", "anatomical_angle_deg", "phi_deg", "connection_valid", "sector_8_name"]:
            self.assertIn(col, ONH3D_MRW_DETAIL_COLUMNS)
        for col in ["slice_id", "side", "scan_angle_deg", "bmo_x_px", "bmo_y_px", "ilm_x_px", "ilm_y_px"]:
            self.assertIn(col, ONH3D_MRW_DETAIL_COLUMNS)
        for col in ["local_area_mm2", "ilm_hit_x_px", "ilm_hit_y_px"]:
            self.assertIn(col, ONH3D_MRA_DETAIL_COLUMNS)

    def test_empty_detail_builders_keep_compat_columns_nullable(self):
        mrw_df = build_empty_onh3d_mrw_detail_df()
        mra_df = build_empty_onh3d_mra_detail_df()
        for col in ["slice_id", "side", "scan_angle_deg", "bmo_x_px", "bmo_y_px"]:
            self.assertIn(col, mrw_df.columns)
            self.assertIn(col, mra_df.columns)
        self.assertEqual(len(mrw_df), 0)
        self.assertEqual(len(mra_df), 0)

    def test_report_meta_includes_required_fields(self):
        meta = build_onh3d_report_meta(build_case())
        self.assertEqual(meta["algorithm_family"], "ONH3D_512")
        self.assertEqual(meta["geometry_model"], "true_3d_onh")
        self.assertEqual(meta["algorithm_version"], "v0")
        self.assertEqual(meta["mra_sector_aggregation"], "sum")

    def test_sector_summary_schema_uses_value_and_aggregation_method(self):
        self.assertIn("value", ONH3D_SECTOR_SUMMARY_COLUMNS)
        self.assertIn("aggregation_method", ONH3D_SECTOR_SUMMARY_COLUMNS)

    def test_adapter_maps_metric_result_without_recomputing(self):
        case = build_case()
        ring_geometry = build_ring_geometry(case.bmo_ring_3d)
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
            theta_ref_deg=123.0,
            sector_8="S8",
            sector_4="S4",
            sector_2="S2",
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
                mrw_um=321.0,
                phi_deg=77.0,
                delta_s_mm=0.5,
                mra_contrib_mm2=0.456,
                theta_ref_deg=123.0,
                sector_8="S8",
                sector_4="S4",
                sector_2="S2",
            )
        ]
        result = ONH3DMetricResult(
            config=ONH3DMetricConfig(),
            ring_geometry=ring_geometry,
            ring_samples=[sample],
            detail=detail,
            sector_summary_8={"S8": ONH3DSectorMetricSummary(8, "S8", 321.0, 321.0, 0.456, 1)},
            sector_summary_4={"S4": ONH3DSectorMetricSummary(4, "S4", 321.0, 321.0, 0.456, 1)},
            sector_summary_2={"S2": ONH3DSectorMetricSummary(2, "S2", 321.0, 321.0, 0.456, 1)},
            MRW_global_mean_um=321.0,
            MRW_global_min_um=321.0,
            MRW_global_low10_mean_um=321.0,
            MRA_global_sum_mm2=0.456,
            valid_sample_count=1,
            total_sample_count=1,
        )

        adapted = adapt_onh3d_metrics_to_stage3_tables(case, result)
        self.assertEqual(adapted["mrw_df"].iloc[0]["mrw_len_um"], 321.0)
        self.assertEqual(adapted["mrw_df"].iloc[0]["phi_deg"], 77.0)
        self.assertEqual(adapted["mra_df"].iloc[0]["local_area_mm2"], 0.456)
        self.assertEqual(adapted["sector_df"].iloc[2]["aggregation_method"], "sum")
        self.assertEqual(adapted["report_meta"]["mra_sector_aggregation"], "sum")


if __name__ == "__main__":
    unittest.main()
