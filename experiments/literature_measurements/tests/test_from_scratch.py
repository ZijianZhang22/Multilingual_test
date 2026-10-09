"""CPU-only plan checks: no HF model files or CUDA required."""
import tempfile
import unittest
from pathlib import Path
from experiments.literature_measurements.from_scratch import (
    build_bootstrap, parse_args, validate_data_metadata,
)


class TestBootstrap(unittest.TestCase):
    def test_default_plan(self):
        a = parse_args(["--dry-run"])
        steps, analysis, data, anchor, adapted = build_bootstrap(a)
        self.assertEqual([s.name for s in steps],
                         ["00_prepare_wikipedia", "00_train_en_then_zh"])
        self.assertTrue(str(anchor).endswith("training/anchor"))
        self.assertTrue(str(adapted).endswith("training/adapted"))
        self.assertIn("--reload_anchor_before_new_stage", steps[1].argv)
        self.assertIn("--anchor", analysis)
        self.assertIn("--adapted", analysis)
        self.assertEqual(a.model_name, "Qwen/Qwen2.5-3B")

    def test_partial_untracked_data_rejected(self):
        a = parse_args(["--dry-run"])
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            (data/"en_train.pt").write_bytes(b"partial")
            with self.assertRaises(RuntimeError):
                validate_data_metadata(data, a)

    def test_invalid_languages(self):
        with self.assertRaises(SystemExit):
            parse_args(["--old-language", "en", "--new-language", "en"])


if __name__ == "__main__":
    unittest.main()
