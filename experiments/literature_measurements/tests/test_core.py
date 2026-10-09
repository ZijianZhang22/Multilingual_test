"""CPU-only regression tests; no weights or Hugging Face downloads needed."""
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.literature_measurements.association import mean_rank, pearson
from experiments.literature_measurements.core import (
    affine_project, ensure_disjoint, fit_affine, linear_cka, mean_pairwise_cosine,
    orthonormal, principal_overlap, with_hidden
)
from experiments.literature_measurements.transfer_probe import accuracy, fit_ridge
from experiments.literature_measurements.retrieval import paired_retrieval, validate_metadata, compute


class TestCore(unittest.TestCase):
    def test_cross_recenter_preserves_own_mean(self):
        h = torch.tensor([[4., 8.], [6., 10.]])
        own = torch.tensor([5., 9.])
        other = torch.tensor([20., 30.])
        q = torch.tensor([[1.], [0.]])
        same = affine_project(h, own, own, q, mode="same")
        cross = affine_project(h, own, other, q, mode="cross")
        recenter = affine_project(h, own, other, q, mode="cross_recenter")
        self.assertTrue(torch.allclose(same, recenter))
        self.assertTrue(torch.allclose(same[:, 1], torch.full((2,), 9.)))
        self.assertTrue(torch.allclose(cross[:, 1], torch.full((2,), 30.)))

    def test_disjoint_ids_and_text(self):
        fit = [{"example_id": "a", "text": "hello"}]
        ensure_disjoint(fit, [{"example_id": "b", "text": "world"}])
        with self.assertRaises(ValueError):
            ensure_disjoint(fit, [{"example_id": "a", "text": "other"}])
        with self.assertRaises(ValueError):
            ensure_disjoint(fit, [{"example_id": "c", "text": "hello"}])

    def test_fit_affine_reports_variance(self):
        torch.manual_seed(11)
        x = torch.randn(80, 8)
        mu, q, meta = fit_affine(x, target_variance=0.98, max_rank=2)
        self.assertEqual(q.shape, (8, 2))
        self.assertFalse(meta["target_reached"])
        self.assertLess(meta["explained_variance"], 0.98)
        self.assertTrue(torch.allclose(q.T @ q, torch.eye(2), atol=1e-5))
        self.assertTrue(torch.allclose(mu, x.mean(0)))

    def test_exact_cross_cosine(self):
        torch.manual_seed(4)
        a = torch.randn(9, 4)
        b = torch.randn(6, 4)
        exact = (torch.nn.functional.normalize(a, dim=-1) @
                 torch.nn.functional.normalize(b, dim=-1).T).mean()
        self.assertAlmostEqual(mean_pairwise_cosine(a, b), exact.item(), places=6)

    def test_cka_and_principal_overlap(self):
        torch.manual_seed(22)
        x = torch.randn(30, 5)
        self.assertAlmostEqual(linear_cka(x, x), 1.0, places=5)
        q = orthonormal(torch.randn(14, 3))
        self.assertAlmostEqual(principal_overlap(q, q), 1.0, places=5)
        with self.assertRaises(ValueError):
            linear_cka(x, x[:-1])

    def test_tuple_hook_preserves_extras(self):
        out = (torch.tensor(1.), "attention_weights")
        result = with_hidden(out, torch.tensor(3.))
        self.assertEqual(result[1], "attention_weights")
        self.assertEqual(result[0].item(), 3.)

    def test_ridge_probe(self):
        x = torch.tensor([[-2., 0.], [-1., 1.], [1., 0.], [2., 1.]])
        y = torch.tensor([0, 0, 1, 1])
        classifier = fit_ridge(x, y, 2, ridge=0.1)
        self.assertEqual(accuracy(classifier, x, y), 1.0)

    def test_paired_retrieval(self):
        query = torch.eye(4)
        self.assertEqual(paired_retrieval(query, query), 1.0)
        self.assertLess(paired_retrieval(query, query.flip(0)), 1.0)

    def test_retrieval_heldout_fit(self):
        fit = {
            "pool": "mean", "source": "residual",
            "languages": ["en", "zh"] * 4,
            "pair_ids": [f"fit:{i}" for i in range(4) for _ in range(2)],
            "features": {"12": torch.randn(8, 4)},
        }
        base = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
        eva = {
            "pool": "mean", "source": "residual",
            "languages": ["en", "zh"] * 4,
            "pair_ids": [f"eval:{i}" for i in range(4) for _ in range(2)],
            "features": {"12": base.repeat_interleave(2, 0)},
        }
        validate_metadata(fit, eva)
        res = compute(fit, eva, 12, "en", "zh", "raw")
        self.assertEqual(res[2], 4)
        self.assertEqual(res[0], 1.0)
        with self.assertRaises(ValueError):
            validate_metadata(fit, {**eva, "pair_ids": fit["pair_ids"]})

    def test_ranks_and_correlations(self):
        self.assertEqual(mean_rank([5, 1, 1, 9]), [3., 1.5, 1.5, 4.])
        self.assertAlmostEqual(pearson([1, 2, 3, 4], [2, 4, 6, 8]), 1.0)
        self.assertIsNone(pearson([1, 2], [3, 4]))


if __name__ == "__main__":
    unittest.main()
