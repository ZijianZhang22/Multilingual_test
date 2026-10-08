"""GPU-free checks for the 1M-token, 2e-5/4e-5 causal runner."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "run_fullzh_lr_mechanisms.py"


class RunFullZHTest(unittest.TestCase):
    def test_dry_run_plan(self):
        with tempfile.TemporaryDirectory() as d:
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), "--dry_run",
                 "--sweep_root", str(Path(d) / "fresh_sweep"),
                 "--analysis_root", str(Path(d) / "analysis")],
                capture_output=True, text=True
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("LR 4e-05", proc.stdout)
            self.assertIn("LR 2e-05", proc.stdout)
            self.assertIn("sweep_learning_rate.py", proc.stdout)
            self.assertIn("--export_lr 4e-05", proc.stdout)
            self.assertIn("--export_lr 2e-05", proc.stdout)
            self.assertIn("build_core_subspaces.py", proc.stdout)
            self.assertIn("run_step7_drift_isr_partition_rescue.py", proc.stdout)
            self.assertNotIn("run_step6_energy_matched_controls.py", proc.stdout)
            self.assertFalse((Path(d) / "analysis").exists())

    def test_dry_run_step6(self):
        with tempfile.TemporaryDirectory() as d:
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), "--dry_run", "--through", "step6",
                 "--sweep_root", str(Path(d) / "fresh_sweep")],
                capture_output=True, text=True
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("run_step6_energy_matched_controls.py", proc.stdout)

    def test_prevent_wrong_data_budget(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = root / "seed0"
            config.mkdir()
            (config / "sweep_config.json").write_text(json.dumps({
                "seed": 0, "model_name": "Qwen/Qwen2.5-7B",
                "old_language": "en", "new_language": "zh",
                "new_train_fraction": 0.2, "lrs": [2e-5, 4e-5],
                "anchor_checkpoint": "/tmp/notused"
            }))
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), "--dry_run",
                 "--sweep_root", str(root)],
                capture_output=True, text=True
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("100% ZH", proc.stderr)


if __name__ == "__main__":
    unittest.main()
