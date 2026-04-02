import unittest
from pathlib import Path


class Stage3SetupDocsTests(unittest.TestCase):
    def test_stage3_requirements_file_lists_runtime_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        req_path = root / "requirements-stage3.txt"
        self.assertTrue(req_path.is_file())
        text = req_path.read_text(encoding="utf-8")
        self.assertIn("numpy", text)
        self.assertIn("pandas", text)
        self.assertIn("openpyxl", text)

    def test_stage3_setup_and_run_scripts_exist(self):
        root = Path(__file__).resolve().parents[1]
        setup_script = root / "scripts" / "setup_stage3_env.sh"
        run_script = root / "scripts" / "run_stage3_512.sh"
        self.assertTrue(setup_script.is_file())
        self.assertTrue(run_script.is_file())
        self.assertIn("requirements-stage3.txt", setup_script.read_text(encoding="utf-8"))
        run_text = run_script.read_text(encoding="utf-8")
        self.assertIn("stage3_main_512.py", run_text)
        self.assertIn("--onh3d-case", run_text)
        self.assertIn("--output-dir", run_text)

    def test_readme_mentions_stage3_512_setup_and_run(self):
        root = Path(__file__).resolve().parents[1]
        readme = root / "README.md"
        self.assertTrue(readme.is_file())
        text = readme.read_text(encoding="utf-8")
        self.assertIn("setup_stage3_env.sh", text)
        self.assertIn("run_stage3_512.sh", text)
        self.assertIn("stage3_main_512.py", text)
        self.assertIn("zuizhong.py", text)


if __name__ == "__main__":
    unittest.main()
