"""CPU-only controls for three-family subspace mechanism experiments."""
import tempfile
import unittest
from pathlib import Path
import torch

from experiments.literature_measurements.subspace_core import (
    REAL_SPACES, choose_donors, get_energy_scales, load_core, projected,
)
from experiments.literature_measurements.subspace_functional_summary import unified


class TestSubspace(unittest.TestCase):
    def test_load_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"core.pt"
            q = torch.eye(8)[:, :2]
            torch.save({"layer": 3, "hidden_dim": 8, "pool": "last",
                        "subspaces": {"drift": q, "transfer": q,
                                      "isr_cov": q, "isr_multiclass": q},
                        "fit_protocol": "probe_train_only_for_drift_isr_center"},
                       p)
            data, spaces = load_core(p, ("drift", "transfer"))
            self.assertEqual(set(spaces), {"drift", "transfer",
                                          "random_drift", "random_transfer"})
            self.assertEqual(spaces["drift"].shape, (8, 2))
            payload = torch.load(p, weights_only=False)
            payload.pop("fit_protocol")
            torch.save(payload, p)
            with self.assertRaises(ValueError):
                load_core(p)
            load_core(p, ("drift",), allow_legacy=True)

    def test_project_direction(self):
        x = torch.tensor([2., 3., 5.])
        q = torch.eye(3)[:, :2]
        self.assertTrue(torch.allclose(projected(x, q), torch.tensor([2., 3., 0.])))

    def test_shrink_only_matching(self):
        d = {
            ("drift", 2): torch.tensor([3., 4.]),
            ("random_drift", 2): torch.tensor([0., 2.]),
            ("transfer", 3): torch.tensor([8., 0.]),
            ("random_transfer", 3): torch.tensor([6., 0.]),
        }
        scales, norms = get_energy_scales(d)
        self.assertAlmostEqual(scales[("drift", 2)], 0.4)
        self.assertAlmostEqual(scales[("transfer", 3)], 0.75)
        self.assertEqual(scales[("random_drift", 2)], 1.)
        for key, scalar in scales.items():
            self.assertLessEqual(scalar, 1.)

    def test_donors_label_and_pair(self):
        target = {"pair_id":"test:0","example_id":"en:0","language":"en","label":1}
        rows = [target,{"pair_id":"test:0","example_id":"zh:0","language":"zh","label":1},
                {"pair_id":"test:1","example_id":"en:1","language":"en","label":1},
                {"pair_id":"test:2","example_id":"zh:2","language":"zh","label":1},
                {"pair_id":"test:3","example_id":"en:3","language":"en","label":0},
                {"pair_id":"test:4","example_id":"zh:4","language":"zh","label":0}]
        chosen = choose_donors(rows,target)
        self.assertEqual(len(chosen),5)
        self.assertEqual(chosen["same_pair_cross_lang"]["pair_id"],"test:0")
        self.assertEqual(chosen["different_pair_cross_lang_same_label"]["label"],1)

    def test_unified_restricts_spaces(self):
        a = unified(
            semantic=[{"subspace":"drift","energy_mode":"matched",
                       "donor_condition":"same_pair_cross_lang",
                       "gold_nll_delta":"0.2","n":"4"}],
            bidir=[{"subspace":"drift","language":"en","direction":"restore",
                    "loss_delta":"-0.1","recovery_fraction":"0.3"},
                   {"subspace":"drift","language":"en","direction":"induce",
                    "loss_delta":"0.1","induced_forgetting_fraction":"0.3"}],
            probe=[], old_language="en", new_language="zh")
        self.assertEqual(len(a),len(REAL_SPACES))
        d = a[0]
        self.assertTrue(d["bidir_available"])
        self.assertEqual(d["semantic_same_pair_gold_nll_delta"],0.2)


if __name__ == "__main__":
    unittest.main()
