from ttach import base
from functools import partial
from ttach import functional as F


class ThreeCrops(base.ImageOnlyTransform):
    """Makes 2 crops for each corner + center crop

    Args:
        crop_height (int): crop height in pixels
        crop_width (int): crop width in pixels
    """

    def __init__(self, crop_height: int, crop_width: int) -> None:
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