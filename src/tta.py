from typing import List

import ttach
from functools import partial
from torch import nn
from ttach import functional as F


class ThreeCrops(ttach.base.ImageOnlyTransform):
    """Makes 2 crops for each corner + center crop."""

    def __init__(self, crop_height: int, crop_width: int) -> None:
        """
        Args:
            crop_height: crop height in pixels
            crop_width: crop width in pixels
        """

        crop_functions = (
            partial(F.crop_lt, crop_h=crop_height, crop_w=crop_width),
            partial(F.crop_rt, crop_h=crop_height, crop_w=crop_width),
            partial(F.center_crop, crop_h=crop_height, crop_w=crop_width),
        )
        super().__init__("crop_fn", crop_functions)

    def apply_aug_image(self, image, crop_fn=None, **kwargs):
        return crop_fn(image)

    def apply_deaug_mask(self, mask, **kwargs):
        raise ValueError("`ThreeCrops` augmentation is not suitable for mask!")

    def apply_deaug_keypoints(self, keypoints, **kwargs):
        raise ValueError("`ThreeCrops` augmentation is not suitable for keypoints!")


def get_tta_model(
    model: nn.Module, crop_method: str, input_size: List[int]
) -> nn.Module:
    """Wraps input model to TTA model.

    Args:
        model: input model without TTA
        crop_method: one of {'resize', 'crop'}. Cropping method of the input images
        input_size: model's input size

    Returns:
        Model with TTA
    """

    transforms = [ttach.HorizontalFlip()]
    if crop_method == "crop":
        transforms.append(
            ThreeCrops(crop_height=input_size[0], crop_width=input_size[1])
        )
    transforms = ttach.Compose(transforms)
    model = ttach.ClassificationTTAWrapper(model, transforms)

    return model
