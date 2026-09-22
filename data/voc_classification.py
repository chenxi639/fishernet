"""VOC image classification. Labels come from ImageSets/Main, not segmentation."""
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

CLASSES = ('aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
           'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor')


def baseline_transform(train=False, image_size=224):
    """Debug/baseline preprocessing; not the final multi-scale patch pipeline.

    Resize whole image to 224 square to retain image-level objects. This warps
    aspect ratio; record this engineering choice separately from paper settings.
    """
    if not isinstance(image_size, int) or image_size < 64:
        raise ValueError('image_size must be an integer >= 64')
    ops = [transforms.Resize((image_size, image_size), antialias=True)]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class VOCClassification(Dataset):
    def __init__(self, root, split='train', transform=None):
        if split not in ('train', 'val', 'trainval'):
            raise ValueError('supported splits: train, val, trainval')
        self.root, self.split = Path(root), split
        self.transform = transform if transform is not None else baseline_transform()
        folder = self.root / 'ImageSets' / 'Main'
        self.ids = (folder / f'{split}.txt').read_text().split()
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError('split must contain unique nonempty image IDs')
        columns = []
        for name in CLASSES:
            rows = [line.split() for line in (folder / f'{name}_{split}.txt').read_text().splitlines() if line.strip()]
            if any(len(row) != 2 for row in rows):
                raise ValueError(f'malformed class file: {name}')
            mapping = {image_id: int(label) for image_id, label in rows}
            if len(mapping) != len(rows) or set(mapping) != set(self.ids):
                raise ValueError(f'duplicate/missing/extra IDs: {name}')
            if not set(mapping.values()) <= {-1, 0, 1}:
                raise ValueError(f'invalid label: {name}')
            columns.append([mapping[i] for i in self.ids])
        self.raw_labels = torch.tensor(columns, dtype=torch.int8).T.contiguous()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        with Image.open(self.root / 'JPEGImages' / f'{image_id}.jpg') as source:
            image = self.transform(source.convert('RGB'))
        raw = self.raw_labels[index].clone()
        return {'image': image, 'target': (raw == 1).float(), 'valid': raw != 0,
                'raw_label': raw, 'image_id': image_id}


def masked_bce(logits, targets, valid):
    """Mean over valid image-class entries; difficult labels contribute zero."""
    if logits.shape != targets.shape or valid.shape != targets.shape or valid.dtype != torch.bool:
        raise ValueError('matching logits/targets/boolean valid shapes required')
    if not valid.any():
        raise ValueError('batch has no valid labels')
    return torch.nn.functional.binary_cross_entropy_with_logits(logits[valid], targets[valid])
