import unittest

import numpy as np

from stage3_onh3d_contract import (
    BMOPlane,
    ILMSurfaceModel,
    ONH3DCase,
    ONH3DQCMeta,
    SectorReference,
    TransformInfo,
    validate_onh3d_case,
)


def build_valid_case():
    return ONH3DCase(
        case_id="CASE-512",
        patient_id="P512",
        laterality="R",
        axial_length=24.3,
        algorithm_version="v0",
        geometry_model="true_3d_onh",
        bmo_ring_3d=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
            ],
            dtype=float,
        ),
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
        transform_info=TransformInfo(
            world_unit="mm",
            voxel_spacing_mm_xyz=(0.01, 0.01, 0.01),
            voxel_to_world=np.eye(4, dtype=float),
        ),
        ilm_surface=ILMSurfaceModel(
            vertices_3d=np.array(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
                dtype=float,
            ),
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


class Stage3ONH3DContractTests(unittest.TestCase):
    def test_validate_onh3d_case_accepts_minimal_valid_case(self):
        validate_onh3d_case(build_valid_case())

    def test_validate_onh3d_case_rejects_invalid_bmo_ring_shape(self):
        case = build_valid_case()
        case.bmo_ring_3d = np.array([0.0, 1.0, 2.0], dtype=float)
        with self.assertRaisesRegex(ValueError, "bmo_ring_3d"):
            validate_onh3d_case(case)

    def test_validate_onh3d_case_rejects_invalid_faces_shape(self):
        case = build_valid_case()
        case.ilm_surface.faces = np.array([0, 1, 2], dtype=int)
        with self.assertRaisesRegex(ValueError, "ilm_surface.faces"):
            validate_onh3d_case(case)

    def test_validate_onh3d_case_rejects_invalid_bmo_plane_vector_shape(self):
        case = build_valid_case()
        case.bmo_plane.origin_3d = np.array([0.0, 0.0], dtype=float)
        with self.assertRaisesRegex(ValueError, "bmo_plane.origin_3d"):
            validate_onh3d_case(case)


if __name__ == "__main__":
    unittest.main()
