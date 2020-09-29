from typing import Tuple, Dict, Optional, Callable, Any, Sequence

import albumentations as albu
import cv2
import numpy as np
import torch
from torch.utils import data as torch_data
import random


class ZindiWheatDataset(torch_data.Dataset):
    def __init__(
        self,
        images: Sequence[str],
        labels: Optional[Sequence[int]] = None,
        preprocess_function: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
        augmentations: Optional[albu.Compose] = None,
        input_shape: Tuple[int, int, int] = (128, 128, 3),
        crop_method: str = "resize",
        augment_label: float = 0.,
    ) -> None:
        self.images = images
        self.labels = labels
        self.preprocess_function = preprocess_function
        self.augmentations = augmentations
        self.input_shape = input_shape
        self.crop_method = crop_method
        self.augment_label = augment_label

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, image_index: int) -> Dict[str, Any]:
        sample = dict()

        # Read data
        sample["image"] = self._read_image(image_index)

        # Read labels
        if self.labels is not None:
            sample = self._read_label(image_index, sample)

        # Crop data
        if self.crop_method is not None:
            sample = self._crop_data(sample)

        # Augment data
        if self.augmentations is not None:
            sample = self._augment_data(sample)

        # Preprocess data
        if self.preprocess_function is not None:
            sample = self._preprocess_data(sample)

        return sample

    def _read_image(self, image_index: int) -> np.ndarray:
        img = cv2.imread(self.images[image_index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _read_label(self, image_index: int, sample: Dict[str, Any]) -> Dict[str, Any]:
        aug_label = self.labels[image_index]

        if self.augment_label > 0:
            p = random.random()
            if p < self.augment_label / 2:
                aug_label = max(0, aug_label - 1)
            elif p < self.augment_label:
                aug_label = min(max(self.labels), aug_label + 1)

        sample["label"] = aug_label
        return sample

    def _crop_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:

        if self.crop_method == "resize":
            aug = albu.Compose(
                [
                    albu.PadIfNeeded(
                        min_height=sample["image"].shape[1] // 2,
                        min_width=sample["image"].shape[1],
                        border_mode=cv2.BORDER_CONSTANT,
                        always_apply=True,
                    ),
                    albu.Resize(
                        height=self.input_shape[0],
                        width=self.input_shape[1],
                        interpolation=cv2.INTER_LINEAR,
                        always_apply=True,
                    ),
                ]
            )
        elif self.crop_method == "crop":
            if self.labels is not None:
                aug = albu.Compose(
                    [
                        albu.PadIfNeeded(
                            min_height=128,
                            min_width=256,
                            border_mode=cv2.BORDER_CONSTANT,
                            always_apply=True,
                        ),
                        albu.RandomCrop(height=128, width=256, always_apply=True),
                    ]
                )
            else:
                aug = albu.Compose(
                    [
                        albu.PadIfNeeded(
                            min_height=128,
                            min_width=256,
                            border_mode=cv2.BORDER_CONSTANT,
                            always_apply=True,
                        ),
                        albu.Resize(
                            height=256,
                            width=512,
                            interpolation=cv2.INTER_LINEAR,
                            always_apply=True,
                        ),
                    ]
                )
        else:
            raise ValueError(f"{self.crop_method} cropping strategy is not available")

        return aug(**sample)

    def _augment_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample = self.augmentations(**sample)
        return sample

    def _preprocess_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample["image"] = self.preprocess_function(sample["image"])
        return sample
