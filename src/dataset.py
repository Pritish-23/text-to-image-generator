# src/dataset.py
# Dataset classes for the project

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import Flowers102
import cv2

FLOWER_NAMES = [
    "pink primrose", "hard-leaved pocket orchid",
    "canterbury bells", "sweet pea", "english marigold",
    "tiger lily", "moon orchid", "bird of paradise",
    "monkshood", "globe thistle", "snapdragon",
    "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe flower", "purple coneflower",
    "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily",
    "pincushion flower", "fritillary", "red ginger",
    "grape hyacinth", "corn poppy",
    "prince of wales feathers", "stemless gentian",
    "artichoke", "sweet william", "carnation",
    "garden phlox", "love in the mist", "mexican aster",
    "alpine sea holly", "ruby-lipped cattleya",
    "cape flower", "great masterwort", "siam tulip",
    "lenten rose", "barbeton daisy", "daffodil",
    "sword lily", "poinsettia", "bolero deep blue",
    "wallflower", "marigold", "buttercup", "oxeye daisy",
    "common dandelion", "petunia", "wild pansy",
    "primula", "sunflower", "pelargonium",
    "bishop of llandaff", "gaura", "geranium",
    "orange dahlia", "pink and yellow dahlia",
    "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush",
    "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower",
    "tree poppy", "gazania", "azalea", "water lily",
    "rose", "thorn apple", "morning glory",
    "passion flower", "lotus", "toad lily",
    "anthurium", "frangipani", "clematis", "hibiscus",
    "columbine", "desert rose", "tree mallow",
    "magnolia", "cyclamen", "watercress", "canna lily",
    "hippeastrum", "bee balm", "ball moss", "foxglove",
    "bougainvillea", "camellia", "mallow",
    "mexican petunia", "bromeliad", "blanket flower",
    "trumpet creeper", "blackberry lily"
]


class FlowerSDDataset(Dataset):
    """Oxford-102 Flowers dataset for Stable Diffusion."""

    def __init__(self, split='train', size=512):
        self.size      = size
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.dataset = Flowers102(
            root='./data', split=split, download=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image        = self.transform(image)
        caption      = (
            f"a photo of a {FLOWER_NAMES[label]} flower, "
            f"highly detailed, vibrant colors, "
            f"nature photography"
        )
        return {'image': image, 'label': label,
                'caption': caption}


class ShapesDataset(Dataset):
    """Custom shapes dataset for CGAN training."""

    def __init__(self, num_samples=10000):
        self.num_samples = num_samples
        self.images, self.labels = self._generate()

    def _generate(self):
        images, labels = [], []
        for _ in range(self.num_samples):
            img   = np.zeros((64, 64), dtype=np.float32)
            label = np.random.randint(0, 3)
            cx    = np.random.randint(20, 44)
            cy    = np.random.randint(20, 44)
            size  = np.random.randint(10, 20)

            if label == 0:
                cv2.circle(img, (cx, cy), size, 1.0, -1)
            elif label == 1:
                cv2.rectangle(
                    img,
                    (cx-size, cy-size),
                    (cx+size, cy+size),
                    1.0, -1
                )
            else:
                pts = np.array([
                    [cx, cy-size],
                    [cx-size, cy+size],
                    [cx+size, cy+size]
                ], np.int32)
                cv2.fillPoly(img, [pts], 1.0)

            img = (img - 0.5) / 0.5
            images.append(img)
            labels.append(label)

        images = np.array(images)[:, np.newaxis, :, :]
        return (torch.FloatTensor(images),
                torch.LongTensor(labels))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
