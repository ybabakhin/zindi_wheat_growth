import albumentations as albu
import cv2


def augs_v1(input_size):
    augmentations = albu.Compose(
        [
            albu.HorizontalFlip(p=0.5),
            albu.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.7
            ),
            albu.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.7,
            ),
            albu.ImageCompression(p=0.3, quality_lower=50, quality_upper=99),
            albu.ToGray(p=0.2),
            albu.HueSaturationValue(p=0.3),
            albu.CLAHE(p=0.3),
            albu.RandomRain(p=0.2),
            albu.RandomShadow(p=0.2),
        ],
        p=1,
    )
    return augmentations


def augs_v2(input_size):
    augmentations = albu.Compose(
        [
            albu.CoarseDropout(
                max_holes=2,
                max_height=512,
                max_width=20,
                min_holes=1,
                min_height=32,
                min_width=10,
                fill_value=255,
                p=0.5,
            ),
            albu.CoarseDropout(
                max_holes=4,
                max_height=128,
                max_width=128,
                min_holes=2,
                min_height=32,
                min_width=32,
                fill_value=0,
                p=0.5,
            ),
            albu.HorizontalFlip(p=0.5),
            albu.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            albu.ImageCompression(p=0.3, quality_lower=55, quality_upper=99),
            albu.ShiftScaleRotate(
                shift_limit=0.3,
                scale_limit=0.3,
                rotate_limit=20,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.8,
            ),
            albu.ToGray(p=0.2),
            albu.HueSaturationValue(p=0.3),
        ],
        p=1,
    )

    return augmentations


class Augmentations:
    _augmentations = {"v1": augs_v1, "v2": augs_v2}

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
