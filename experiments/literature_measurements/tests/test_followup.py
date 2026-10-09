"""CPU-safe tests for followup layout and no-override guards."""
import tempfile
import unittest
from pathlib import Path
from argparse import Namespace

from experiments.literature_measurements.run_followup import plan


class TestFollowup(unittest.TestCase):
    def test_plan_does_not_touch_training_or_full_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"source"
            for p in ("training/anchor","training/adapted","analysis/core","analysis"):
                (root/p).mkdir(parents=True,exist_ok=True)
            for p in ("training/anchor/config.json","training/adapted/config.json",
                      "analysis/core/core_subspaces.pt","analysis/aligned_test.jsonl"):
                (root/p).write_text("mock")
            (root/"wiki").mkdir()
            for lang in ("en","zh"):
                (root/"wiki"/f"{lang}_val.pt").write_text("mock")
            a=Namespace(source=str(root),out=str(Path(tmp)/"new_followup"),
                        semantic_targets_per_lang=30,max_blocks=128,
                        batch_size=2,seed=2027)
            out,steps=plan(a)
            self.assertEqual(len(steps),6)
            self.assertEqual(len({str(x[2]) for x in steps}),6)
            self.assertTrue(all(str(x[2]).startswith(str(out)) for x in steps))
            self.assertTrue(all("--core_file" in x[1] for x in steps))


if __name__=="__main__":
    unittest.main()
