import albumentations as albu
import cv2


def base(input_size):
    augmentations = albu.Compose(
        [
            albu.HorizontalFlip(p=0.5),
            albu.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            albu.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5,
            ),
            albu.ImageCompression(p=0.3, quality_lower=50, quality_upper=99),
        ],
        p=1,
    )
    return augmentations


def base_v1(input_size):
    augmentations = albu.Compose(
        [
            albu.HorizontalFlip(p=0.5),
            albu.CoarseDropout(
                max_holes=2,
                max_height=128,
                max_width=3,
                min_holes=1,
                min_height=32,
                min_width=2,
                fill_value=255,
                p=0.5,
            ),
            albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
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


class Augmentations:
    _augmentations = {"base": base, "base_v1": base_v1}

    @classmethod
    def names(cls):
        return sorted(cls._augmentations.keys())

    @classmethod
    def get(cls, name):
        """
        Access to augmentation strategies
        Args:
            name (str): augmentation strategy name
        Returns:
            callable: function to build augmentation strategy
        """

        return cls._augmentations.get(name)
