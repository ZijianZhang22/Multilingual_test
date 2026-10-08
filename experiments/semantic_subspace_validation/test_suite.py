"""CPU-only smoke tests for 7B semantic validation helpers."""
import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import drift_partition
from probes_retrieval import probe_fit_predict,retrieval_all
from causal_semantics import donor_for
from run_existing_7b import check_ckpt

class ValidationTests(unittest.TestCase):
    def test_partition(self):
        qd=torch.eye(64)
        qi=qd[:,:32]
        partitions,angles=drift_partition(qd,qi)
        self.assertGreater(float(angles[:16].mean()),float(angles[-16:].mean()))
        self.assertLess(float((partitions['top32'].T@partitions['bottom32']).abs().max()),1e-5)

    def test_probe(self):
        rng=np.random.default_rng(23)
        tr=rng.normal(size=(300,8))
        te=rng.normal(size=(100,8))
        r=probe_fit_predict(tr,te,(tr[:,0]>0).astype(int),(te[:,0]>0).astype(int))
        self.assertGreater(r['accuracy'],0.85)

    def test_retrieval(self):
        with tempfile.TemporaryDirectory() as d:
            n,dim=30,64
            x=torch.nn.functional.normalize(torch.randn(n,dim),dim=1)
            feat={'layer':'20','features':{'20':torch.cat([x,x+0.001*torch.randn_like(x)])},'languages':['en']*n+['zh']*n}
            rows=[{'language':lang,'pair_id':str(i),'label':i%3,'split':'aligned','premise':f'topic {i}','hypothesis':f'item {i}'}
                  for lang in ('en','zh') for i in range(n)]
            result=retrieval_all({'test':torch.eye(dim)},feat,rows,Path(d),n)
            self.assertEqual(len(result),2)
            self.assertGreater(result[0]['recall1'],0.95)

    def test_donor(self):
        rows=[{'language':lang,'label':label,'example_id':str(lang)+str(label)}
              for lang in ('en','zh','fr') for label in range(3)]
        a=rows[0]
        b=donor_for(rows,a,'nli')
        self.assertEqual(a['language'],b['language'])
        self.assertNotEqual(a['label'],b['label'])
        b=donor_for(rows,a,'language_id')
        self.assertNotEqual(a['language'],b['language'])
        self.assertEqual(a['label'],b['label'])

    def test_sharded_checkpoint_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'config.json').write_text('{}')
            (p/'model-00001-of-00002.safetensors').write_bytes(b'x')
            (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'model-00001-of-00002.safetensors','b':'model-00002-of-00002.safetensors'}}))
            self.assertFalse(check_ckpt(p))
            (p/'model-00002-of-00002.safetensors').write_bytes(b'y')
            self.assertTrue(check_ckpt(p))

if __name__=='__main__':unittest.main()
