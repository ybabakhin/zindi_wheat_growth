import albumentations as albu


def augs_v1():
    augmentations = albu.Compose(
        [
            albu.HorizontalFlip(p=0.5),
            albu.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.3
            ),
            albu.ImageCompression(p=0.2, quality_lower=55, quality_upper=99),
            albu.ShiftScaleRotate(
                shift_limit=0.2, scale_limit=0.2, rotate_limit=10, p=0.5
            ),
        ],
        p=1,
    )

    return augmentations


class Augmentations:
    _augmentations = {"v1": augs_v1}

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
