from torch.utils.data import Dataset
from albumentations import Resize, Compose, PadIfNeeded
import cv2


class ZindiWheatDataset(Dataset):
    def __init__(
        self,
        images,
        labels=None,
        preprocess_function=None,
        augmentations=None,
        input_shape=(128, 128, 3),
        crop_function="resize",
    ):
        self.images = images
        self.labels = labels
        self.preprocess_function = preprocess_function
        self.augmentations = augmentations
        self.input_shape = input_shape
        self.crop_function = crop_function

    def __len__(self):
        return len(self.images)

    def __getitem__(self, image_index):
        sample = {}

        # Read data
        sample["image"] = self._read_image(image_index)

        # Read labels
        if self.labels is not None:
            sample = self._read_label(image_index, sample)

        # Crop data
        if self.crop_function is not None:
            sample = self._crop_data(sample)

        # Augment data
        if self.augmentations is not None:
            sample = self._augment_data(sample)

        # Preprocess data
        if self.preprocess_function is not None:
            sample = self._preprocess_data(sample)

        return sample

    def _read_image(self, image_index):
        img = cv2.imread(self.images[image_index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _read_label(self, image_index, sample):
        sample["label"] = self.labels[image_index]
        return sample

    def _crop_data(self, sample):

        if self.crop_function == "resize":
            aug = Compose(
                [
                    PadIfNeeded(
                        min_height=sample["image"].shape[1] // 2,
                        min_width=sample["image"].shape[1],
                        border_mode=cv2.BORDER_CONSTANT,
                        always_apply=True,
                    ),
                    Resize(
                        height=self.input_shape[0],
                        width=self.input_shape[1],
                        interpolation=cv2.INTER_AREA,
                        always_apply=True,
                    ),
                ]
            )
        else:
            raise ValueError(f"{self.crop_function} cropping strategy is not available")

        return aug(**sample)

    def _augment_data(self, sample):
        sample = self.augmentations(**sample)
        return sample

    def _preprocess_data(self, sample):
        sample["image"] = self.preprocess_function(sample["image"])
        return sample
