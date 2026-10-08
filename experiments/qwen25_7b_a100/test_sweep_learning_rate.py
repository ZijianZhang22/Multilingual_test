import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "sweep_learning_rate.py"
spec = importlib.util.spec_from_file_location("sweep_learning_rate", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SweepCLI(unittest.TestCase):
    def call(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *args],
                              text=True, capture_output=True)

    def test_dry_run_default(self):
        p = self.call("--dry_run")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.count("reload same EN anchor"), 5)
        for lr in ["1e-05", "2e-05", "4e-05", "5e-05", "6e-05"]:
            self.assertIn(f"lr={lr}", p.stdout)

    def test_invalid_values(self):
        for args in [
            ("--dry_run", "--lrs", "0"),
            ("--dry_run", "--lrs", "1e-5", "1e-5"),
            ("--dry_run", "--new_train_fraction", "2"),
            ("--dry_run", "--export_lr", "9e-5"),
        ]:
            p = self.call(*args)
            self.assertNotEqual(p.returncode, 0)

    def test_dry_run_export_lr(self):
        p = self.call("--dry_run", "--export_lr", "2e-5")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("export selected LR=2e-05 only", p.stdout)

    def test_extend_existing_lr_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            original = {"lrs": [1e-5, 2e-5, 4e-5], "seed": 0}
            module.verify_or_extend_config(path, original)
            extended = {"lrs": [1e-5, 2e-5, 4e-5, 5e-5, 6e-5], "seed": 0}
            module.verify_or_extend_config(path, extended)
            self.assertEqual(json.loads(path.read_text())["lrs"], extended["lrs"])
            with self.assertRaises(ValueError):
                module.verify_or_extend_config(path, {"lrs": extended["lrs"], "seed": 1})

    def test_csv_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for lr in [1e-5, 2e-5]:
                folder = path / f"lr_{module.canonical_lr(lr)}"
                folder.mkdir()
                row = {
                    "learning_rate": lr, "seed": 0,
                    "old_language": "en", "new_language": "zh",
                    "anchor_en_loss": 2.5, "adapted_en_loss": 2.6,
                    "forgetting_loss_delta": 0.1,
                    "anchor_zh_loss": 2.8, "adapted_zh_loss": 2.7,
                    "new_language_gain": 0.1, "zh_tokens_seen": 100,
                    "zh_optimizer_steps": 3, "checkpoint_saved": False,
                }
                (folder / "metrics.json").write_text(json.dumps(row))
            module.save_table(path, [1e-5, 2e-5, 4e-5])
            with (path / "lr_forgetting_summary.csv").open() as fp:
                got = list(csv.DictReader(fp))
            self.assertEqual(len(got), 2)
            self.assertEqual(
                [float(r["learning_rate"]) for r in got], [1e-5, 2e-5]
            )


if __name__ == "__main__":
    unittest.main()
