"""CPU-only tests of stage ordering, dry-run plan and archive integrity."""
import tempfile
import unittest
from pathlib import Path

from experiments.literature_measurements.run_all import (
    build_plan, make_archive, parse_args, safe_filename,
)
from experiments.literature_measurements.split_probe import split_rows


class TestRunner(unittest.TestCase):
    def args(self, mode):
        return parse_args(["--anchor","/tmp/anchor_not_loaded",
                           "--adapted","/tmp/adapted_not_loaded",
                           "--mode",mode,"--out","/tmp/lm_experiment_test"])

    def test_pilot_order_and_outputs(self):
        plan = build_plan(self.args("pilot"))
        self.assertEqual([s.name for s in plan],
                         ["01_core","02_aligned_test","03_semantic",
                          "04_bidirectional","18_unified_summary"])
        all_outputs = [str(p) for step in plan for p in step.outputs]
        self.assertEqual(len(all_outputs), len(set(all_outputs)))
        self.assertIn("core_subspaces.pt", all_outputs[0])

    def test_full_dependencies(self):
        plan = build_plan(self.args("full"))
        names = [s.name for s in plan]
        self.assertIn("06_affine", names)
        self.assertIn("10_aligned_cka", names)
        self.assertIn("12_transfer_probe", names)
        self.assertIn("17_retrieval", names)
        self.assertLess(names.index("15_aligned_fit"), names.index("17_retrieval"))
        self.assertEqual(names[-1], "18_unified_summary")
        self.assertEqual(len(names), len(set(names)))

    def test_probe_split(self):
        rows = [{"example_id":"a","text":"hello","split":"probe_train"},
                {"example_id":"b","text":"world","split":"probe_test"}]
        fit, ev = split_rows(rows)
        self.assertEqual(len(fit),1)
        self.assertEqual(len(ev),1)
        with self.assertRaises(ValueError):
            split_rows([rows[0],{**rows[0],"split":"probe_test"}])

    def test_result_archive_excludes_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"measurements"
            out.mkdir()
            (out/"results.csv").write_text("ok")
            (out/"feature.pt").write_bytes(b"secret")
            archive = make_archive(out)
            import tarfile
            with tarfile.open(archive) as t:
                names = t.getnames()
            self.assertEqual(names, ["measurements/results.csv"])

    def test_filename(self):
        self.assertEqual(safe_filename("step_name-01"), "step_name-01")


if __name__ == "__main__":
    unittest.main()
