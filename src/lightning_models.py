import glob
import logging
from argparse import Namespace
from typing import Union, Tuple, Dict, Any, Sequence, List, Optional

import cnn_finetune
import efficientnet_pytorch
import hydra
import numpy as np
import omegaconf
import os
import pandas as pd
import pytorch_lightning as pl
import torch
from dataclasses import dataclass
from pytorch_lightning.core.step_result import EvalResult
from sklearn import metrics
from torch import nn
from torch.optim import optimizer
from torch.utils import data as torch_data
from torchvision import transforms

from src import augmentations
from src import dataset
from src import utils

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MixupOutput:
    """Data class wrapping mixup augmentation output"""

    data: torch.Tensor
    labels: torch.Tensor
    shuffled_labels: torch.Tensor
    lam: torch.Tensor


class LitWheatModel(pl.LightningModule):
    def __init__(
        self,
        hparams: Optional[Union[dict, Namespace, str]] = None,
        hydra_cfg: Optional[omegaconf.DictConfig] = None,
    ) -> None:
        super(LitWheatModel, self).__init__()

        self.cfg = hydra_cfg
        self.multipliers = np.array(self.cfg.data_mode.rmse_multipliers)

        # Number of classes in bad labels does not equal to the number of classes in good labels
        init_model_num_classes = self.cfg.data_mode.num_classes
        if self.cfg.training.pretrain_dir != "":
            last_path = os.path.join(self.cfg.training.pretrain_dir, "last.ckpt")
            if os.path.exists(last_path):
                pretrain_path = last_path
            else:
                pretrain_path = glob.glob(
                    os.path.join(self.cfg.training.pretrain_dir, "*.ckpt")
                )[0]
            checkpoint = torch.load(pretrain_path, map_location="cpu")
            init_model_num_classes = (
                checkpoint["state_dict"]
                .get(
                    "model._fc.weight",
                    checkpoint["state_dict"].get("model._classifier.weight"),
                )
                .size()[0]
            )

        if self.cfg.model.architecture_name.startswith("efficientnet"):
            self.model = efficientnet_pytorch.EfficientNet.from_pretrained(
                self.cfg.model.architecture_name, num_classes=init_model_num_classes
            )

            self.model._avg_pooling = nn.AdaptiveAvgPool2d(1)
            self.model._dropout = nn.Dropout(self.cfg.model.dropout)

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        else:
            self.model = cnn_finetune.make_model(
                self.cfg.model.architecture_name,
                num_classes=init_model_num_classes,
                pretrained=True,
                dropout_p=self.cfg.model.dropout,
                pool=nn.AdaptiveAvgPool2d(1),
            )

            mean = self.model.original_model_info.mean
            std = self.model.original_model_info.std

        self.preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

        if self.cfg.model.regression:
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    def setup(self, stage: str = "fit") -> None:
        train = pd.read_csv(self.cfg.data_mode.train_csv)
        train = utils.preprocess_df(train, data_dir=self.cfg.data_mode.data_dir)
        train["label"] = train["growth_stage"]
        train = train[train.label_quality >= self.cfg.data_mode.label_quality].copy()

        # Regression labels
        if self.cfg.model.regression:
            train["label"] = train["label"].astype("float32")
        # Bad quality labels
        elif self.cfg.data_mode.label_quality == 1:
            train["label"] = train["growth_stage"] - 1
            train.loc[train["label_quality"] == 1, "fold"] = -1
        # Good quality labels
        else:
            train.loc[train["growth_stage"] < 6, "label"] = (
                train.loc[train["growth_stage"] < 6, "label"] - 2
            )
            train.loc[train["growth_stage"] > 6, "label"] = (
                train.loc[train["growth_stage"] > 6, "label"] - 3
            )
        if self.cfg.data_mode.pseudolabels_path == "":
            self.df_train = train[train.fold != self.cfg.training.fold].reset_index(
                drop=True
            )
            self.df_valid = train[
                (train.fold == self.cfg.training.fold) & (train.label_quality == 2)
            ].reset_index(drop=True)
        else:
            pseudo = pd.read_csv(self.cfg.data_mode.pseudolabels_path)
            pseudo = utils.preprocess_df(pseudo, data_dir=self.cfg.data_mode.data_dir)

            pseudo["growth_stage"] = pseudo["growth_stage"].apply(round).astype(int)
            pseudo.loc[pseudo["growth_stage"] == 6, "growth_stage"] = 7
            pseudo["label"] = pseudo["growth_stage"]

            pseudo.loc[pseudo["growth_stage"] < 6, "label"] = (
                pseudo.loc[pseudo["growth_stage"] < 6, "label"] - 2
            )
            pseudo.loc[pseudo["growth_stage"] > 6, "label"] = (
                pseudo.loc[pseudo["growth_stage"] > 6, "label"] - 3
            )

            self.df_train = pseudo.copy()
            self.df_valid = pseudo.copy()

        logger.info(
            f"Length of the train: {len(self.df_train)}. Length of the validation: {len(self.df_valid)}"
        )

    def train_dataloader(self) -> torch_data.DataLoader:
        augs = augmentations.Augmentations.get(self.cfg.training.augmentations)(
            *self.cfg.model.input_size
        )

        train_dataset = dataset.ZindiWheatDataset(
            images=self.df_train.path.values,
            labels=self.df_train.label.values,
            preprocess_function=self.preprocess,
            augmentations=augs,
            input_shape=(self.cfg.model.input_size[0], self.cfg.model.input_size[1], 3),
            crop_method=self.cfg.model.crop_method,
            augment_label=self.cfg.training.label_augmentation,
        )

        train_loader = torch_data.DataLoader(
            train_dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.general.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(
        self
    ) -> Union[torch_data.DataLoader, List[torch_data.DataLoader]]:
        valid_dataset = dataset.ZindiWheatDataset(
            images=self.df_valid.path.values,
            labels=self.df_valid.label.values,
            preprocess_function=self.preprocess,
            augmentations=None,
            input_shape=(self.cfg.model.input_size[0], self.cfg.model.input_size[1], 3),
            crop_method=self.cfg.model.crop_method,
        )

        valid_loader = torch_data.DataLoader(
            valid_dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.general.num_workers,
            shuffle=False,
            pin_memory=True,
        )

        return valid_loader

    def configure_optimizers(
        self,
    ) -> Optional[
        Union[
            optimizer.Optimizer,
            Sequence[optimizer.Optimizer],
            Dict,
            Sequence[Dict],
            Tuple[List, List],
        ]
    ]:
        num_train_steps = len(self.train_dataloader()) * self.cfg.training.max_epochs
        optimizer = hydra.utils.instantiate(
            self.cfg.optimizer, params=self.parameters()
        )

        try:
            lr_scheduler = hydra.utils.instantiate(
                self.cfg.scheduler, optimizer=optimizer, T_0=num_train_steps
            )
        except hydra.errors.HydraException:
            lr_scheduler = hydra.utils.instantiate(
                self.cfg.scheduler, optimizer=optimizer
            )

        scheduler = {
            "scheduler": lr_scheduler,
            "interval": self.cfg.scheduler.step,
            "monitor": self.cfg.scheduler.monitor,
        }

        return [optimizer], [scheduler]

    @staticmethod
    def mixup(
        data: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2
    ) -> MixupOutput:
        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]
        shuffled_labels = labels[indices]

        lam = np.random.beta(alpha, alpha, size=len(indices))
        lam = np.maximum(lam, 1 - lam)
        lam = lam.reshape(lam.shape + (1,) * (len(data.shape) - 1))
        lam = torch.Tensor(lam).cuda()

        data = lam * data + (1 - lam) * shuffled_data

        mixup_output = MixupOutput(
            data=data, labels=labels, shuffled_labels=shuffled_labels, lam=lam
        )

        return mixup_output

    @staticmethod
    def rand_bbox(
        height: int, width: int, lam: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = (width * cut_rat).astype(int)
        cut_h = (height * cut_rat).astype(int)

        # Uniform
        cx = np.random.randint(width, size=len(lam))
        cy = np.random.randint(height, size=len(lam))

        bbx1 = np.clip(cx - cut_w // 2, 0, width)
        bby1 = np.clip(cy - cut_h // 2, 0, height)
        bbx2 = np.clip(cx + cut_w // 2, 0, width)
        bby2 = np.clip(cy + cut_h // 2, 0, height)

        return bbx1, bby1, bbx2, bby2

    def cutmix(
        self, data: torch.Tensor, labels: torch.Tensor, alpha: float = 0.4
    ) -> MixupOutput:
        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]
        shuffled_labels = labels[indices]

        lam = np.random.beta(alpha, alpha, size=len(indices))
        lam = np.maximum(lam, 1 - lam)

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(
            height=data.size()[2], width=data.size()[3], lam=lam
        )
        for idx in range(data.shape[0]):
            data[idx, :, bbx1[idx] : bbx2[idx], bby1[idx] : bby2[idx]] = shuffled_data[
                idx, :, bbx1[idx] : bbx2[idx], bby1[idx] : bby2[idx]
            ]
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
        lam = torch.Tensor(lam).cuda()

        cutmix_output = MixupOutput(
            data=data, labels=labels, shuffled_labels=shuffled_labels, lam=lam
        )

        return cutmix_output

    def mixup_cutmix_criterion(
        self, preds: torch.Tensor, mixup_output: MixupOutput
    ) -> torch.Tensor:
        non_reduction_loss = nn.CrossEntropyLoss(reduction="none")
        loss = mixup_output.lam * non_reduction_loss(preds, mixup_output.labels) + (
            1 - mixup_output.lam
        ) * non_reduction_loss(preds, mixup_output.shuffled_labels)

        return torch.mean(loss)

    def _model_step(
        self, batch: Dict[str, torch.Tensor], mixup: bool = False, cutmix: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        images = batch["image"]
        labels = batch["label"]

        if mixup:
            mixup_output = self.mixup(
                data=images, labels=labels, alpha=self.cfg.training.mixup
            )
            preds = self(mixup_output.data)
            loss = self.mixup_cutmix_criterion(preds=preds, mixup_output=mixup_output)
        elif cutmix:
            cutmix_output = self.cutmix(
                data=images, labels=labels, alpha=self.cfg.training.cutmix
            )
            preds = self(cutmix_output.data)
            loss = self.mixup_cutmix_criterion(preds=preds, mixup_output=cutmix_output)
        else:
            preds = self(images)
            if self.cfg.model.regression:
                preds = preds.view(-1)
            loss = self.criterion(preds, labels)

        return preds, loss

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        _, loss = self._model_step(
            batch,
            mixup=self.cfg.training.mixup > 0,
            cutmix=self.cfg.training.cutmix > 0,
        )
        tensorboard_logs = {"train_loss": loss}

        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        preds, loss = self._model_step(batch)
        if not self.cfg.model.regression:
            preds = torch.softmax(preds, dim=1)

        return {"preds": preds, "step_val_loss": loss}

    def validation_epoch_end(
        self, outputs: Union[EvalResult, List[EvalResult]]
    ) -> Dict[str, Any]:
        if self.cfg.model.regression:
            preds = np.concatenate([x["preds"].cpu().detach().numpy() for x in outputs])
        else:
            preds = np.vstack([x["preds"].cpu().detach().numpy() for x in outputs])

        avg_loss = torch.stack([x["step_val_loss"] for x in outputs]).mean().item()

        if not self.cfg.model.regression:
            preds = np.sum(preds * self.multipliers, axis=-1)
        preds = np.clip(preds, min(self.multipliers), max(self.multipliers))

        labels = self.df_valid.growth_stage.values
        if len(labels) == len(preds):
            rmse = np.sqrt(metrics.mean_squared_error(preds, labels))
        else:
            rmse = 0

        tensorboard_logs = {
            "val_loss": avg_loss,
            "val_rmse": rmse,
            "step": self.current_epoch,
        }

        return {
            "val_loss": avg_loss,
            "val_rmse": rmse,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }
