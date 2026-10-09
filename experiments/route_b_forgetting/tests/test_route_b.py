import importlib.util
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('route_b', ROOT / 'route_b.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


class RouteBTests(unittest.TestCase):
    def test_exact_split_dedup(self):
        self.assertEqual(mod.document_split(' hello world '), mod.document_split('hello world'))
        splits = {mod.document_split(f'article {i}')[1] for i in range(200)}
        self.assertEqual(splits, {'train', 'val', 'test'})

    def test_matched_schedule(self):
        self.assertEqual(mod.arm_languages('mixed', 8).count('zh'), 4)
        self.assertEqual(mod.arm_languages('zh_only', 8), ['zh'] * 8)
        self.assertEqual(mod.arm_languages('en_only', 8), ['en'] * 8)
        with self.assertRaises(ValueError):
            mod.arm_languages('mixed', 3)

    def test_paired_document_bootstrap(self):
        before = np.array([1., 2., 3., 4.])
        docs = ['a', 'a', 'b', 'c']
        r = mod.bootstrap_delta(before, before + .5, docs)
        self.assertAlmostEqual(r['delta'], .5)
        np.testing.assert_allclose(r['ci95'], [.5, .5])
        r = mod.bootstrap_delta(before, before, docs)
        self.assertEqual(r['ci95'], [0., 0.])

    def test_budget(self):
        args = mod.parser().parse_args(['run'])
        self.assertEqual(args.steps * args.micro_batch * args.grad_accum * args.block_size, 1048576)

class TrainingIntegrationTests(unittest.TestCase):
    def test_real_optimizer_and_matched_baselines(self):
        """Tiny random model checks actual torch training, not scientific forgetting."""
        try:
            import torch
            from transformers import LlamaConfig, LlamaForCausalLM
        except ImportError:
            self.skipTest('torch/transformers unavailable')
        import contextlib
        import io
        import json
        import tempfile
        from unittest.mock import patch
        config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=128)
        class Tokenizer:
            def save_pretrained(self, path):
                (Path(path) / 'tokenizer_test.json').write_text('{}')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'data').mkdir()
            rng = np.random.default_rng(1)
            for lang in ('en', 'zh'):
                np.savez(root / 'data' / f'{lang}.npz', train=rng.integers(1, 64, (4, 16)),
                         val=rng.integers(1, 64, (4, 16)), test=rng.integers(1, 64, (4, 16)))
            for arm in mod.ARMS:
                args = mod.parser().parse_args(['train', '--cpu', '--arm', arm, '--out', str(root),
                    '--steps', '2', '--block-size', '16', '--micro-batch', '1', '--grad-accum', '2',
                    '--eval-blocks', '4', '--eval-batch', '2', '--eval-marks', '--save-marks', '1'])
                with patch('transformers.AutoModelForCausalLM.from_pretrained', side_effect=lambda *a, **k: LlamaForCausalLM(config)), \
                     patch('transformers.AutoTokenizer.from_pretrained', return_value=Tokenizer()), \
                     contextlib.redirect_stdout(io.StringIO()):
                    mod.train(args)
                metrics = json.loads((root / arm / 'metrics.json').read_text())
                self.assertEqual([m['step'] for m in metrics], [0, 1, 2])
                self.assertEqual(metrics[-1]['tokens_seen'], 64)
                self.assertTrue((root / arm / 'step_000001' / 'model.safetensors').exists())
                self.assertTrue((root / arm / 'step_000002' / 'model.safetensors').exists())
                if arm == 'mixed':
                    self.assertEqual(metrics[-1]['language_tokens_seen'], dict(en=32, zh=32))
                if arm == 'zh_only':
                    baseline = metrics[0]['losses']
                else:
                    self.assertEqual(baseline, metrics[0]['losses'])


if __name__ == '__main__':
    unittest.main()
