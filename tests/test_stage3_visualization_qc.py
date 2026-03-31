import importlib.util
import shutil
import unittest
import uuid
from pathlib import Path

from stage3_shared import build_legacy_cloud_from_shared, build_stage3_shared_structure
from stage3_visualization import save_stage3_qc_3d_views


def load_stage3_main_module():
    root = Path(__file__).resolve().parents[1]
    target = root / "zuizhong.py"
    assert target.is_file()
    spec = importlib.util.spec_from_file_location("stage3_main_module", str(target))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_stage2_case_payload():
    slices = []
    for scan_index in range(1, 13):
        slices.append(
            {
                "scan_index": scan_index,
                "slice_stem": f"slice_{scan_index:02d}",
                "image_width": 20,
                "image_height": 100,
                "angle_deg": float((scan_index - 1) * 15.0),
                "image_shape": [100, 20],
                "full_ilm_px": [[float(x), 20.0 + 0.2 * scan_index + 0.02 * (x - 10) ** 2] for x in range(0, 20)],
                "bmo_left_px": [5.0, 40.0],
                "bmo_right_px": [14.0, 40.0],
                "cutoff_left_px": [4.0, 34.0],
                "cutoff_right_px": [15.0, 34.0],
                "rnfl_effective_lower_px": [[4.0, 60.0], [9.0, 62.0], [15.0, 60.0]],
                "review_status": "approved",
                "source_flags": {"stage2_final": True},
            }
        )
    return {
        "stage2_schema_version": "v1",
        "case_id": "CASE-QC",
        "patient_id": "PQC",
        "laterality": "R",
        "image_width": 20,
        "image_height": 100,
        "x_center": 10.0,
        "y_center": 50.0,
        "n_slices": 12,
        "axial_length": 24.0,
        "slices": slices,
    }


class Stage3VisualizationQCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_stage3_main_module()

    def setUp(self):
        workspace_tmp = Path(__file__).resolve().parents[1] / "tests_tmp"
        workspace_tmp.mkdir(exist_ok=True)
        self.root = workspace_tmp / f"visualization_qc_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_save_stage3_qc_3d_views_writes_pngs_and_manifest(self):
        shared_case = build_stage3_shared_structure(build_stage2_case_payload(), {})
        final_cloud = build_legacy_cloud_from_shared(shared_case)
        aligned_cloud = self.module.align_to_bmo_bfp(final_cloud)

        written = save_stage3_qc_3d_views(str(self.root), aligned_cloud)

        expected = {
            "stage3_qc_3d_top.png",
            "stage3_qc_3d_oblique.png",
            "stage3_qc_3d_side.png",
            "qc_3d_manifest.json",
        }
        self.assertEqual({Path(path).name for path in written}, expected)
        for path in written:
            self.assertTrue(Path(path).is_file(), path)


if __name__ == "__main__":
    unittest.main()
