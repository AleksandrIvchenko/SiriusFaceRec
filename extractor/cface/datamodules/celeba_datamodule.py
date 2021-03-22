from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from albumentations import (
    Compose,
    HorizontalFlip,
    Normalize,
    Resize,
)
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CelebA

from cface.datamodules.base_datamodule import BaseDataModule
from cface.utils import LandmarksProcessor


class CelebADataset(Dataset):
    def __init__(
            self,
            root,
            target_type: str = 'identity',
            transform: Optional[List] = None,
            download: bool = False,
        ):
        images_path = root / 'celeba' / 'img_align_celeba'
        identity_path = root / 'celeba' / 'identity_CelebA.txt'
        landmarks_path = root / 'celeba' / 'list_landmarks_align_celeba.txt'
        identity_file = open(identity_path)
        landmarks_file = open(landmarks_path)
        landmarks_file.readline().split()
        landmarks_file.readline().split()

        self.filenames = list(p for p in images_path.glob('*.jpg'))
        n_images = len(self.filenames)

        self.labels = dict()
        for i in range(n_images):
            filename, label = identity_file.readline().split()
            self.labels[filename] = int(label)

        self.landmarks = dict()
        for i in range(n_images):
            info = landmarks_file.readline().split()
            filename = info.pop(0)
            info = list(map(int, info))
            self.landmarks[filename] = np.array([
                [info[0], info[1]],
                [info[2], info[3]],
                [info[4], info[5]],
                [info[6], info[7]],
                [info[8], info[9]],
            ])

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        filename_2 = filename.name
        label = self.labels[filename_2]
        ldm = self.landmarks[filename_2]
        image = cv2.imread(str(filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = LandmarksProcessor.cut_face(
            img=image,
            box=None,
            ldm=ldm,
        )

        if self.transform:
            image = self.transform(image=image)['image']

        return image, label


class CelebADataModule(BaseDataModule):
    def setup(
            self,
            val_ratio: float = 0.1,
            new_size: Tuple[int, int] = (128, 128),
            download: bool = False,
        ):
        data_transforms = Compose([
            Resize(new_size[0], new_size[1]),
            Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            #HorizontalFlip(), TODO find how to flip landmarks
            ToTensorV2(),
        ])
        full_dataset = CelebADataset(
            root=self.data_path,
            target_type='identity',
            transform=data_transforms,
            download=download,
        )

        full_size = len(full_dataset)
        val_size = int(val_ratio * full_size)
        train_size = full_size - val_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset=full_dataset,
            lengths=[train_size, val_size],
        )


if __name__ == '__main__':
    dm = CelebADataModule(
        data_path=Path('./data/celeba'),
        batch_size=64,
        num_workers=4,
    )
    dm.setup()
    dl = dm.train_dataloader()
    print(len(dl))

