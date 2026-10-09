import argparse
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import compare_activations as core
import run_suite as runner


class HeatmapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.x = np.random.default_rng(42).normal(size=(5, 3, 12))
        self.meta = dict(model='before', tokenizer='before', dtype='float32', pool='last', pooling_version=2,
                         max_length=128, module_layers=[1], input_sha256='hash',
                         rows=[dict(id=str(i), text=f'Sample {i}') for i in range(5)],
                         token_ids=[[1, 2, 3] for _ in range(5)])
        self.save('before', self.x)

    def save(self, name, hidden, token_ids=None):
        meta = dict(self.meta, model=name)
        if token_ids is not None:
            meta['token_ids'] = token_ids
        path = self.root / f'{name}.npz'
        np.savez_compressed(path, hidden=hidden, loss=np.ones(5), counts=np.full(5, 2),
                            metadata=np.array(json.dumps(meta)), module_1_up_proj=hidden[:, 1])
        return path

    def compare(self, name, dest='comparison'):
        path = self.root / dest
        core.compare(argparse.Namespace(before=self.root / 'before.npz',
                                        after=self.root / f'{name}.npz', out=path, top_k=8))
        return path

    def test_metrics(self):
        np.testing.assert_allclose(core.cosine(self.x, self.x), 1)
        np.testing.assert_allclose(core.cosine(self.x, -self.x), -1)
        self.assertAlmostEqual(core.cka(self.x[:, 0], self.x[:, 0]), 1)
        q, _ = np.linalg.qr(np.random.default_rng(9).normal(size=(12, 12)))
        self.assertAlmostEqual(core.cka(self.x[:, 0], 3 * self.x[:, 0] @ q), 1)
        self.assertTrue(np.isnan(core.cka(np.zeros((5, 12)), self.x[:, 0])))

    def test_identity_and_gallery(self):
        self.save('after', self.x)
        comp = self.compare('after')
        with (comp / 'per_sample_layer.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 15)
        self.assertTrue(all(float(r['relative_drift']) == 0 for r in rows))
        self.assertEqual(len(list(comp.glob('*.png'))), 10)
        out = self.root.parent / (self.root.name + '_gallery')
        # Place the gallery in the existing temp root, with a pair subdirectory.
        pair = self.root / 'pair'
        pair.mkdir()
        import shutil
        shutil.copytree(comp, pair / 'comparison')
        np.savez(pair / 'before.npz', hidden=self.x)
        plan = [(dict(pair='test', before='before', after='after'), pair, [])]
        archive = runner.gallery(self.root, plan)
        with zipfile.ZipFile(archive) as z:
            self.assertIn('index.html', z.namelist())
            self.assertIn('pair/comparison/00_overview.png', z.namelist())
            self.assertFalse(any(s.endswith('.npz') for s in z.namelist()))

    def test_multi_pool_gallery_archive(self):
        self.save('after', self.x)
        comp = self.compare('after')
        import shutil
        plan = []
        pair = self.root / 'multi'
        pair.mkdir()
        (pair / 'before.log').write_text('shared extraction')
        for pool in core.POOLS:
            directory = pair / pool
            directory.mkdir()
            shutil.copytree(comp, directory / 'comparison')
            np.savez(directory / 'before.npz', hidden=self.x)
            plan.append((dict(pair='multi', before='before', after='after', pool=pool), directory, []))
        archive = runner.gallery(self.root, plan)
        with zipfile.ZipFile(archive) as z:
            self.assertIn('multi/before.log', z.namelist())
            for pool in core.POOLS:
                self.assertIn(f'multi/{pool}/comparison/diagnostics.json', z.namelist())
                self.assertIn(f'multi/{pool}/comparison/group_layer_summary.csv', z.namelist())
            self.assertFalse(any(name.endswith('.npz') for name in z.namelist()))

    def test_sign_reversal(self):
        self.save('after', -self.x)
        comp = self.compare('after')
        with (comp / 'per_sample_layer.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(all(abs(float(r['relative_drift']) - 2) < 1e-10 for r in rows))
        self.assertTrue(all(abs(float(r['cosine']) + 1) < 1e-10 for r in rows))

    def test_reject_changed_input(self):
        self.save('after', self.x, token_ids=[[1, 2, 4]] * 5)
        with self.assertRaisesRegex(ValueError, 'token_ids'):
            self.compare('after')

    def test_resume_checks_file_metadata(self):
        self.save('after', self.x)
        item = dict(before='before', after='after', tokenizer='before', input_sha256='hash',
                    pool='last', dtype='float32', max_length=128, module_layers=[1])
        self.assertTrue(runner.valid_extraction(self.root / 'after.npz', item, 'after'))
        self.assertFalse(runner.valid_extraction(self.root / 'before.npz', item, 'after'))
        item['input_sha256'] = 'changed'
        self.assertFalse(runner.valid_extraction(self.root / 'after.npz', item, 'after'))

    def test_pooling_content_and_punctuation(self):
        class Tokenizer:
            all_special_ids = [0, 9]
            def decode(self, ids, **kwargs):
                return {0: '<bos>', 1: 'Hello', 2: ',', 3: ' world', 4: '.', 5: '!', 6: ' ', 7: '你好', 8: 'word.', 9: '<eos>'}[ids[0]]
        tok = Tokenizer()
        self.assertEqual(core.pooling_indices(tok, [0, 1, 2, 3, 4, 5, 9], 'mean'), [1, 2, 3, 4, 5])
        self.assertEqual(core.pooling_indices(tok, [0, 1, 2, 3, 4, 5, 9], 'last_nonpunct'), [3])
        self.assertEqual(core.pooling_indices(tok, [0, 7, 4, 9], 'last_nonpunct'), [1])
        self.assertEqual(core.pooling_indices(tok, [0, 8, 4], 'last_nonpunct'), [1])
        self.assertEqual(core.pooling_indices(tok, [0, 1, 9], 'last'), [2])
        with self.assertRaises(ValueError):
            core.pooling_indices(tok, [0, 4, 5, 9], 'last_nonpunct')

    def test_pair_and_custom_plan(self):
        args = argparse.Namespace(inputs=None, limit=100, before=None, after=None,
                pair='all', tokenizer=None, out=str(self.root), pool='last', dtype='float32',
                max_length=128, layers=None, device='cpu', top_k=64)
        _, _, plan = runner.build_plan(args)
        self.assertEqual(len(plan), 3)
        self.assertEqual(plan[0][0]['before'], 'TinyLlama/TinyLlama_v1.1')
        self.assertEqual(plan[0][1], self.root / 'tiny_zh')
        self.assertEqual(len(plan[0][2]), 3)
        args.pool = 'all'
        _, _, pooled = runner.build_plan(args)
        self.assertEqual(len(pooled[0][2]), 5)  # two forwards, three comparisons
        self.assertIn('/mean/before.npz', ' '.join(pooled[0][2][2][1]))
        self.assertIn('/last_nonpunct/before.npz', ' '.join(pooled[0][2][3][1]))
        args.pool = 'last'
        args.before = '/anchor'
        with self.assertRaises(ValueError):
            runner.build_plan(args)
        args.after = '/adapted'
        _, _, plan = runner.build_plan(args)
        self.assertEqual(plan[0][0]['pair'], 'custom')


if __name__ == '__main__':
    unittest.main()
