from typing import Callable, Optional, List

import albumentations as albu
import cv2


def base(input_h: int, input_w: int) -> albu.Compose:
    """Base augmentation strategy.

    Args:
        input_h: image height
        input_w: image width

    Returns:
        List of augmentations
    """

    augmentations = albu.Compose(
        [
            albu.HorizontalFlip(p=0.5),
            albu.CoarseDropout(
                max_holes=2,
                max_height=input_h // 2,
                max_width=input_w // 128 + 1,
                min_holes=1,
                min_height=input_h // 8,
                min_width=input_w // 128,
                fill_value=255,
                p=0.5,
            ),
            albu.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            albu.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5,
            ),
        ],
        p=1,
    )
    return augmentations


def hard(input_h: int, input_w: int) -> albu.Compose:
    """Hard augmentation strategy.

    Args:
        input_h: image height
        input_w: image width

    Returns:
        List of augmentations
    """

    augmentations = albu.Compose(
        [
            albu.HorizontalFlip(p=0.5),
            albu.CoarseDropout(
                max_holes=2,
                max_height=input_h // 2,
                max_width=input_w // 128 + 1,
                min_holes=1,
                min_height=input_h // 8,
                min_width=input_w // 128,
                fill_value=255,
                p=0.7,
            ),
            albu.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.7
            ),
            albu.ShiftScaleRotate(
                shift_limit=0.3,
                scale_limit=0.3,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.7,
            ),
        ],
        p=1,
    )
    return augmentations


class Augmentations:
    """Augmentations factory."""

    _augmentations = {"base": base, "hard": hard}

    @classmethod
    def names(cls) -> List[str]:
        return sorted(cls._augmentations.keys())

    @classmethod
    def get(cls, name: str) -> Optional[Callable[[int, int], albu.Compose]]:
        """Access to augmentation strategies

        Args:
            name: augmentation strategy name
        Returns:
            A function to build augmentation strategy
        """

        return cls._augmentations.get(name)
