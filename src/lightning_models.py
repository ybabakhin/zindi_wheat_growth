import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch.nn as nn
from cnn_finetune import make_model as make_pretrained_model
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from src.dataset import ZindiWheatDataset
from src.augmentations import Augmentations
from sklearn.metrics import mean_squared_error
from src.utils import preprocess_df
import hydra


class LitWheatModel(pl.LightningModule):
    def __init__(self, hparams=None, hydra_cfg=None):
        super(LitWheatModel, self).__init__()

        self.cfg = hydra_cfg

        # Number of classes in bad labels does not equal to the number of classes in good labels
        init_model_num_classes = self.cfg.data_mode.num_classes
        if self.cfg.training.pretrain_path != "":
            checkpoint = torch.load(self.cfg.training.pretrain_path)
            init_model_num_classes = (
                checkpoint["state_dict"]
                .get("model._fc.weight", checkpoint["state_dict"]["model._classifier.weight"])
                .shape[0]
            )

        if self.cfg.training.architecture_name.startswith("efficientnet"):
            self.model = EfficientNet.from_pretrained(
                self.cfg.training.architecture_name, num_classes=init_model_num_classes
            )

            self.model._avg_pooling = nn.AdaptiveAvgPool2d(1)
            self.model._dropout = nn.Dropout(self.cfg.training.dropout)

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        else:
            self.model = make_pretrained_model(
                self.cfg.training.architecture_name,
                num_classes=init_model_num_classes,
                pretrained=True,
                dropout_p=self.cfg.training.dropout,
                pool=nn.AdaptiveAvgPool2d(1),
            )

            mean = self.model.original_model_info.mean
            std = self.model.original_model_info.std

        self.preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

        if self.cfg.training.regression:
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, x):
        x = self.model(x)
        return x

    def setup(self, stage="fit"):
        if self.cfg.data_mode.pseudolabels:
            pass
        else:
            train = pd.read_csv(self.cfg.data_mode.train_csv)
            train = preprocess_df(train, data_dir=self.cfg.data_mode.data_dir)
            train["label"] = train["growth_stage"]
            train = train[train.label_quality >= self.cfg.data_mode.label_quality].copy()

            # Regression labels
            if self.cfg.training.regression:
                train["label"] = train["label"].astype("float32")
            # Bad quality labels
            elif self.cfg.data_mode.label_quality == 1:
                train["label"] = train["growth_stage"] - 1
            # Good quality labels
            else:
                train.loc[train["growth_stage"] < 6, "label"] = (
                    train.loc[train["growth_stage"] < 6, "label"] - 2
                )
                train.loc[train["growth_stage"] > 6, "label"] = (
                    train.loc[train["growth_stage"] > 6, "label"] - 3
                )

            self.df_train = train[train.fold != self.cfg.training.fold].reset_index(drop=True)
            self.df_valid = train[
                (train.fold == self.cfg.training.fold) & (train.label_quality == 2)
            ].reset_index(drop=True)

    def train_dataloader(self):
        augs = Augmentations.get(self.cfg.training.augmentations)(self.cfg.training.input_size)

        self.train_dataset = ZindiWheatDataset(
            images=self.df_train.path.values,
            labels=self.df_train.label.values,
            preprocess_function=self.preprocess,
            augmentations=augs,
            input_shape=(self.cfg.training.input_size, self.cfg.training.input_size, 3),
            crop_function="resize",
        )

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        self.valid_dataset = ZindiWheatDataset(
            images=self.df_valid.path.values,
            labels=self.df_valid.label.values,
            preprocess_function=self.preprocess,
            augmentations=None,
            input_shape=(self.cfg.training.input_size, self.cfg.training.input_size, 3),
            crop_function="resize",
        )

        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers,
            shuffle=False,
            pin_memory=True,
        )

        return valid_loader

    def configure_optimizers(self):
        num_train_steps = len(self.train_dataloader()) * self.cfg.training.max_epochs
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, params=self.parameters())

        try:
            lr_scheduler = hydra.utils.instantiate(
                self.cfg.scheduler, optimizer=optimizer, T_0=num_train_steps
            )
        except hydra.errors.HydraException:
            lr_scheduler = hydra.utils.instantiate(self.cfg.scheduler, optimizer=optimizer)

        scheduler = {
            "scheduler": lr_scheduler,
            "interval": self.cfg.scheduler.step,
            "monitor": self.cfg.scheduler.monitor,
        }

        return [optimizer], [scheduler]

    def _model_step(self, batch):
        images = batch["image"]
        labels = batch["label"]

        preds = self(images)
        if self.cfg.training.regression:
            preds = preds.view(-1)

        loss = self.criterion(preds, labels)
        return preds, loss

    def training_step(self, batch, batch_idx):
        _, loss = self._model_step(batch)
        tensorboard_logs = {"train_loss": loss}

        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        preds, loss = self._model_step(batch)
        if not self.cfg.training.regression:
            preds = torch.softmax(preds, dim=1)

        return {"preds": preds, "step_val_loss": loss}

    def validation_epoch_end(self, outputs):
        if self.cfg.training.regression:
            preds = np.concatenate([x["preds"].cpu().detach().numpy() for x in outputs])
        else:
            preds = np.vstack([x["preds"].cpu().detach().numpy() for x in outputs])

        avg_loss = torch.stack([x["step_val_loss"] for x in outputs]).mean().item()

        multipliers = np.array(self.cfg.data_mode.rmse_multipliers)
        if not self.cfg.training.regression:
            preds = np.sum(preds * multipliers, axis=-1)
        preds = np.clip(preds, min(multipliers), max(multipliers))

        labels = self.df_valid.growth_stage.values
        if len(labels) == len(preds):
            rmse = np.sqrt(mean_squared_error(preds, labels))
        else:
            rmse = 0

        tensorboard_logs = {"val_loss": avg_loss, "val_rmse": rmse, "step": self.current_epoch}

        return {
            "val_loss": avg_loss,
            "val_rmse": rmse,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }
