"""python step03_verify.py --report runs/step03_verification.json"""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from data import VOCClassification, CLASSES, masked_bce
from metrics import voc_ap, classification_map

ROOT = Path(__file__).resolve().parent
REPORT = {}


class MetricTests(unittest.TestCase):
    def test_hand_computed(self):
        self.assertEqual(voc_ap([1,1,-1], [3,2,1]), 1.)
        self.assertAlmostEqual(voc_ap([-1,1,1], [3,2,1]), 2/3)
        self.assertAlmostEqual(voc_ap([1,-1,1], [3,2,1]), 5/6)
        self.assertAlmostEqual(voc_ap([0,1,-1,1], [100,3,2,1]), 5/6)
        self.assertTrue(np.isnan(voc_ap([-1,0], [2,1])))
        self.assertEqual(voc_ap([1,-1], [1,1]), 1.)
        self.assertEqual(voc_ap([-1,1], [1,1]), .5)
        result = classification_map(np.array([[1,-1],[-1,1]]), np.array([[2,2],[1,1]]))
        self.assertEqual(result['map'], .75)

    def test_independent_definition(self):
        # Slow direct implementation of the official precision envelope rule,
        # independently integrating at each positive recall increment.
        rng = np.random.default_rng(123)
        for _ in range(100):
            labels = rng.choice([-1,0,1], size=30)
            labels[0] = 1
            scores = rng.normal(size=30)
            ranked = [int(labels[i]) for i in sorted(range(30), key=lambda i: -scores[i]) if labels[i] != 0]
            p = [ranked[:i+1].count(1)/(i+1) for i in range(len(ranked))]
            expected = sum(max(p[i:]) for i,y in enumerate(ranked) if y == 1)/ranked.count(1)
            self.assertAlmostEqual(voc_ap(labels, scores), expected)

    def test_invalid_metric_inputs(self):
        for labels,scores in [([1],[float('nan')]), ([2],[1]), ([],[]), ([1],[1,2])]:
            with self.assertRaises(ValueError):
                voc_ap(labels,scores)

    def test_loss_mask(self):
        logits = torch.zeros(1,3,requires_grad=True)
        targets = torch.tensor([[1.,0.,0.]])
        valid = torch.tensor([[True,True,False]])
        loss = masked_bce(logits,targets,valid)
        self.assertAlmostEqual(loss.item(), np.log(2), places=6)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.tensor([[-.25,.25,0.]]))
        with self.assertRaises(ValueError):
            masked_bce(logits,targets,torch.zeros_like(valid))

    def test_dataset_alignment_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main = root/'ImageSets/Main'
            main.mkdir(parents=True)
            (root/'JPEGImages').mkdir()
            (main/'train.txt').write_text('b\na\n')
            for c in CLASSES:
                (main/f'{c}_train.txt').write_text('a -1\nb 1\n')
            Image.new('L',(20,30),128).save(root/'JPEGImages/b.jpg')
            ds = VOCClassification(root)
            self.assertTrue((ds.raw_labels[0] == 1).all())
            self.assertTrue((ds.raw_labels[1] == -1).all())
            self.assertEqual(ds[0]['image'].shape,(3,224,224))
            (main/'cat_train.txt').write_text('a -1\na 1\n')
            with self.assertRaises(ValueError):
                VOCClassification(root)


def inspect_data(root, samples, workers):
    train, val, union = [VOCClassification(root,s) for s in ('train','val','trainval')]
    assert not set(train.ids)&set(val.ids)
    assert set(train.ids)|set(val.ids) == set(union.ids)
    REPORT['splits'] = {d.split:len(d) for d in (train,val,union)}
    REPORT['class_order'] = list(CLASSES)
    REPORT['counts'] = {}
    for ds in (train,val):
        missing = [i for i in ds.ids if not (root/'JPEGImages'/f'{i}.jpg').is_file() or not (root/'Annotations'/f'{i}.xml').is_file()]
        assert not missing, missing[:5]
        raw = ds.raw_labels.numpy()
        REPORT['counts'][ds.split] = {c:{'positive':int((raw[:,j]==1).sum()),'negative':int((raw[:,j]==-1).sum()),'ignored':int((raw[:,j]==0).sum())} for j,c in enumerate(CLASSES)}
        # Evenly spaced samples include the last item; decode images and check
        # image-level class labels independently from XML object annotations.
        indices = np.unique(np.linspace(0,len(ds)-1,min(samples,len(ds)),dtype=int))
        for idx in indices:
            item = ds[int(idx)]
            assert item['image'].shape == (3,224,224) and torch.isfinite(item['image']).all()
            objects = ET.parse(root/'Annotations'/f'{item["image_id"]}.xml').getroot().findall('object')
            expected = []
            for c in CLASSES:
                matches = [o for o in objects if o.findtext('name') == c]
                expected.append(1 if any(o.findtext('difficult','0') == '0' for o in matches) else 0 if matches else -1)
            assert item['raw_label'].tolist() == expected, item['image_id']
        REPORT[f'{ds.split}_decoded_xml_checked'] = len(indices)
    batch = next(iter(DataLoader(train,batch_size=4,num_workers=workers,shuffle=False)))
    assert batch['image'].shape == (4,3,224,224)
    assert batch['target'].shape == batch['valid'].shape == (4,20)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logits = torch.zeros(4,20,device=device,requires_grad=True)
    loss = masked_bce(logits,batch['target'].to(device),batch['valid'].to(device))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    REPORT.update(batch_shape=list(batch['image'].shape), loss_device=device, masked_bce=float(loss.detach()), workers=workers)
    # Oracle scores are only an evaluation plumbing test, never model results.
    oracle = classification_map(val.raw_labels.numpy(),val.raw_labels.numpy())
    assert oracle['map'] == 1.
    REPORT['oracle_metric_sanity_only'] = oracle['map']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT/'datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012')
    parser.add_argument('--samples',type=int,default=32)
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--report',type=Path,default=ROOT/'runs/step03_verification.json')
    args = parser.parse_args()
    if args.samples <= 0 or args.workers < 0:
        parser.error('samples must be positive; workers must be nonnegative')
    torch.set_num_threads(2)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MetricTests))
    REPORT.update(unit_tests=result.testsRun,unit_tests_passed=result.wasSuccessful(),passed=False)
    try:
        if not result.wasSuccessful():
            raise RuntimeError('unit tests failed')
        inspect_data(args.root,args.samples,args.workers)
        REPORT['passed'] = True
    except Exception as exc:
        REPORT['error'] = str(exc)
        raise
    finally:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(REPORT,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in REPORT.items() if k not in ('counts','class_order')},indent=2))
