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
from stage3_onh3d_metrics import (
    ONH3DLocalFrame,
    ONH3DLocalMetric,
    ONH3DMetricConfig,
    ONH3DMetricResult,
    ONH3DRingSample,
    ONH3DSectorMetricSummary,
    aggregate_global_metrics,
    aggregate_sector_metrics,
    build_ring_geometry,
    build_ring_samples,
    compute_local_metric,
    compute_low_fraction_mean,
    compute_onh3d_metrics,
    compute_local_tangent_periodic,
    resample_ring_equal_arclength,
)
from stage3_onh3d_report_adapter import adapt_onh3d_metrics_to_stage3_tables


def build_case(*, clockwise_positive=True):
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
    return ONH3DCase(
        case_id="CASE-512",
        patient_id="P512",
        laterality="R",
        axial_length=24.0,
        algorithm_version="v1",
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
            clockwise_positive=clockwise_positive,
        ),
        transform_info=TransformInfo(world_unit="mm"),
        ilm_surface=ILMSurfaceModel(
            vertices_3d=np.array(
                [
                    [-2.0, -2.0, 1.0],
                    [2.0, -2.0, 1.0],
                    [2.0, 2.0, 1.0],
                    [-2.0, 2.0, 1.0],
                ],
                dtype=float,
            ),
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=int),
        ),
        qc_meta=ONH3DQCMeta(
            bmo_ring_closed=True,
            bmo_ring_sampling_count=16,
            ilm_surface_vertex_count=4,
            ilm_surface_face_count=2,
            bmo_plane_fit_rms_mm=0.0,
            notes=[],
        ),
    )


def build_sample(theta=0.0, sector_8="A8", sector_4="A4", sector_2="A2", delta_s=0.5):
    frame = ONH3DLocalFrame(
        origin_3d=np.array([0.0, 0.0, 0.0], dtype=float),
        tangent_3d=np.array([1.0, 0.0, 0.0], dtype=float),
        reference_normal_3d=np.array([0.0, 0.0, 1.0], dtype=float),
        plane_x_axis_3d=np.array([1.0, 0.0, 0.0], dtype=float),
        plane_y_axis_3d=np.array([0.0, 1.0, 0.0], dtype=float),
        plane_normal_3d=np.array([0.0, -1.0, 0.0], dtype=float),
    )
    return ONH3DRingSample(
        sample_idx=0,
        arc_length_s_mm=0.0,
        delta_s_mm=delta_s,
        bmo_point_3d=np.array([0.0, 0.0, 0.0], dtype=float),
        bmo_tangent_3d=np.array([1.0, 0.0, 0.0], dtype=float),
        theta_ref_deg=theta,
        sector_8=sector_8,
        sector_4=sector_4,
        sector_2=sector_2,
        local_frame=frame,
    )


class Stage3ONH3DMetricsTests(unittest.TestCase):
    def test_low10_mean_uses_ceil(self):
        values = np.arange(1.0, 129.0)
        self.assertEqual(compute_low_fraction_mean(values), 7.0)
        self.assertEqual(compute_low_fraction_mean(np.arange(1.0, 10.0)), 1.0)
        self.assertEqual(compute_low_fraction_mean(np.arange(1.0, 12.0)), 1.5)

    def test_mra_sector_aggregation_is_sum(self):
        metrics = [
            ONH3DLocalMetric(0, True, None, np.zeros(3), np.ones(3), np.ones(3), 10.0, 90.0, 1.0, 1.0, 0.0, "S8", "S4", "S2"),
            ONH3DLocalMetric(1, True, None, np.zeros(3), np.ones(3), np.ones(3), 20.0, 90.0, 1.0, 2.0, 0.0, "S8", "S4", "S2"),
            ONH3DLocalMetric(2, True, None, np.zeros(3), np.ones(3), np.ones(3), 30.0, 90.0, 1.0, 3.0, 0.0, "S8", "S4", "S2"),
        ]
        summary = aggregate_sector_metrics(metrics, ONH3DMetricConfig())
        self.assertEqual(summary[8]["S8"].mra_sum_mm2, 6.0)

    def test_sector_reference_affects_labels_only(self):
        sample_a = build_sample(theta=10.0, sector_8="A8", sector_4="A4", sector_2="A2")
        sample_b = build_sample(theta=250.0, sector_8="B8", sector_4="B4", sector_2="B2")
        connection = {"ilm_point_3d": np.array([0.0, 1.0, 0.0], dtype=float)}
        metric_a = compute_local_metric(sample_a, connection)
        metric_b = compute_local_metric(sample_b, connection)
        self.assertEqual(metric_a.mrw_um, metric_b.mrw_um)
        self.assertEqual(metric_a.phi_deg, metric_b.phi_deg)
        self.assertEqual(metric_a.mra_contrib_mm2, metric_b.mra_contrib_mm2)
        self.assertNotEqual(metric_a.theta_ref_deg, metric_b.theta_ref_deg)
        self.assertNotEqual(metric_a.sector_8, metric_b.sector_8)

    def test_periodic_ring_resampling_and_tangent(self):
        case = build_case()
        ring = build_ring_geometry(case.bmo_ring_3d)
        points = resample_ring_equal_arclength(ring, 128)
        self.assertEqual(points.shape, (128, 3))
        self.assertGreater(ring.total_arc_length_mm, 0.0)
        step_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        self.assertTrue(np.all(np.isfinite(step_lengths)))
        self.assertLess(np.std(step_lengths), 0.05)
        tangent_0 = compute_local_tangent_periodic(points, 0)
        tangent_last = compute_local_tangent_periodic(points, 127)
        self.assertTrue(np.all(np.isfinite(tangent_0)))
        self.assertTrue(np.all(np.isfinite(tangent_last)))
        self.assertGreater(np.linalg.norm(tangent_0), 0.0)
        self.assertGreater(np.linalg.norm(tangent_last), 0.0)

    def test_phi_deg_semantics(self):
        sample = build_sample()
        metric_90 = compute_local_metric(sample, {"ilm_point_3d": np.array([0.0, 1.0, 0.0], dtype=float)})
        self.assertAlmostEqual(metric_90.phi_deg, 90.0, places=6)

        metric_45 = compute_local_metric(sample, {"ilm_point_3d": np.array([1.0, 1.0, 0.0], dtype=float)})
        self.assertAlmostEqual(metric_45.phi_deg, 45.0, places=6)

    def test_invalid_samples_excluded_from_global_and_sector_aggregation(self):
        valid = ONH3DLocalMetric(0, True, None, np.zeros(3), np.ones(3), np.ones(3), 10.0, 90.0, 1.0, 1.0, 0.0, "S8", "S4", "S2")
        invalid = ONH3DLocalMetric(1, False, "bad", np.zeros(3), None, None, np.nan, np.nan, 1.0, 0.0, 0.0, "S8", "S4", "S2")
        global_summary = aggregate_global_metrics([valid, invalid], ONH3DMetricConfig())
        sector_summary = aggregate_sector_metrics([valid, invalid], ONH3DMetricConfig())
        self.assertEqual(global_summary["MRW_global_mean_um"], 10.0)
        self.assertEqual(global_summary["MRW_global_min_um"], 10.0)
        self.assertEqual(global_summary["MRW_global_low10_mean_um"], 10.0)
        self.assertEqual(global_summary["MRA_global_sum_mm2"], 1.0)
        self.assertEqual(sector_summary[8]["S8"].valid_sample_count, 1)
        self.assertEqual(sector_summary[8]["S8"].mra_sum_mm2, 1.0)

    def test_adapter_consumes_metric_result_without_recomputing_core_values(self):
        case = build_case()
        ring_geometry = build_ring_geometry(case.bmo_ring_3d)
        sample = build_sample(theta=123.0, sector_8="S8", sector_4="S4", sector_2="S2")
        metric = ONH3DLocalMetric(
            sample_idx=0,
            is_valid=True,
            invalid_reason=None,
            bmo_point_3d=np.array([0.0, 0.0, 0.0], dtype=float),
            ilm_point_3d=np.array([0.0, 1.0, 0.0], dtype=float),
            connection_vec_3d=np.array([0.0, 1.0, 0.0], dtype=float),
            mrw_um=222.0,
            phi_deg=88.0,
            delta_s_mm=0.5,
            mra_contrib_mm2=0.789,
            theta_ref_deg=123.0,
            sector_8="S8",
            sector_4="S4",
            sector_2="S2",
        )
        result = ONH3DMetricResult(
            config=ONH3DMetricConfig(),
            ring_geometry=ring_geometry,
            ring_samples=[sample],
            detail=[metric],
            sector_summary_8={"S8": ONH3DSectorMetricSummary(8, "S8", 222.0, 222.0, 0.789, 1)},
            sector_summary_4={"S4": ONH3DSectorMetricSummary(4, "S4", 222.0, 222.0, 0.789, 1)},
            sector_summary_2={"S2": ONH3DSectorMetricSummary(2, "S2", 222.0, 222.0, 0.789, 1)},
            MRW_global_mean_um=222.0,
            MRW_global_min_um=222.0,
            MRW_global_low10_mean_um=222.0,
            MRA_global_sum_mm2=0.789,
            valid_sample_count=1,
            total_sample_count=1,
        )
        adapted = adapt_onh3d_metrics_to_stage3_tables(case, result)
        self.assertEqual(adapted["mrw_df"].iloc[0]["mrw_len_um"], 222.0)
        self.assertEqual(adapted["mrw_df"].iloc[0]["phi_deg"], 88.0)
        self.assertEqual(adapted["mra_df"].iloc[0]["local_area_mm2"], 0.789)
        self.assertEqual(
            adapted["sector_df"][adapted["sector_df"]["metric_name"] == "MRA_local_area_mm2"].iloc[0]["value"],
            0.789,
        )

    def test_compute_onh3d_metrics_returns_valid_metrics_for_planar_ilm_case(self):
        case = build_case()
        result = compute_onh3d_metrics(case)
        self.assertEqual(result.total_sample_count, 128)
        self.assertGreater(result.valid_sample_count, 0)
        self.assertTrue(np.isfinite(result.MRW_global_mean_um))
        self.assertTrue(np.isfinite(result.MRW_global_min_um))
        self.assertTrue(np.isfinite(result.MRW_global_low10_mean_um))
        self.assertGreater(result.MRA_global_sum_mm2, 0.0)
        self.assertEqual(len(result.ring_samples), 128)
        self.assertEqual(len(result.detail), 128)
        self.assertTrue(result.sector_summary_8)


if __name__ == "__main__":
    unittest.main()
