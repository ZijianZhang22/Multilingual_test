"""CPU-only CLI smoke tests: no model downloads, CUDA, or external datasets."""
import subprocess
import sys
import unittest
from pathlib import Path

RUNNER = Path(__file__).with_name("run_7b_a100.py")


class SevenBA100RunnerTest(unittest.TestCase):
    def invoke(self, *extra):
        return subprocess.run(
            [sys.executable, str(RUNNER), "--dry_run", *extra],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False,
        )

    def test_full_command_plan(self):
        proc = self.invoke("--through", "step6")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("layer=23", proc.stdout)
        self.assertIn("train_7b_a100.py", proc.stdout)
        self.assertIn("build_core_subspaces.py", proc.stdout)
        self.assertIn("run_step7_drift_isr_partition_rescue.py", proc.stdout)
        self.assertIn("run_step6_energy_matched_controls.py", proc.stdout)
        self.assertIn("--ks 16 32", proc.stdout)
        self.assertIn("--n_random 8", proc.stdout)

    def test_training_only(self):
        proc = self.invoke("--through", "training")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("train_7b_a100.py", proc.stdout)
        self.assertNotIn("run_step7_drift_isr_partition_rescue.py", proc.stdout)

    def test_invalid_relative_layer(self):
        proc = self.invoke("--relative_layer", "0")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("relative_layer", proc.stderr)

    def test_invalid_rank(self):
        proc = self.invoke("--rank", "32")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("rank", proc.stderr)

    def test_separate_seed_output(self):
        proc = self.invoke("--seed", "2", "--through", "training")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("seed2", proc.stdout)
        self.assertIn("--seed 2", proc.stdout)


if __name__ == "__main__":
    unittest.main()
