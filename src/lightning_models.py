import os
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch.nn as nn
from torch import optim
from cnn_finetune import make_model as make_pretrained_model
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from src.dataset import ZindiWheatDataset
from src.augmentations import Augmentations
from sklearn.metrics import mean_squared_error
from src.utils import preprocess_df


class LitWheatModel(pl.LightningModule):
    def __init__(self, hydra_cfg):
        super(LitWheatModel, self).__init__()

        self.cfg = hydra_cfg

        if self.cfg.training.architecture_name.startswith("efficientnet"):

            self.model = EfficientNet.from_pretrained(
                self.cfg.training.architecture_name,
                num_classes=self.cfg.training.num_classes,
            )

            self.model._avg_pooling = nn.AdaptiveAvgPool2d(1)
            self.model._dropout = nn.Dropout(self.cfg.training.dropout)

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        else:
            self.model = make_pretrained_model(
                self.cfg.training.architecture_name,
                num_classes=self.cfg.training.num_classes,
                pretrained=True,
                dropout_p=self.cfg.training.dropout,
                pool=nn.AdaptiveAvgPool2d(1),
                # input_size=self.cfg.training.input_size,
            )

            mean = self.model.original_model_info.mean
            std = self.model.original_model_info.std

        self.preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, x):
        x = self.model(x)
        return x

    def prepare_data(self):
        if self.cfg.training.pseudolabels:
            pass
        else:
            train = pd.read_csv(self.cfg.training.train_csv)
            train = preprocess_df(train, data_dir=self.cfg.training.data_dir)

            train["label"] = train["growth_stage"] - 1

            self.df_train = train[train.fold != self.cfg.training.fold].reset_index(
                drop=True
            )
            self.df_valid = train[
                (train.fold == self.cfg.training.fold) & (train.label_quality == 2)
            ].reset_index(drop=True)

    def train_dataloader(self):
        augs = Augmentations.get(self.cfg.training.augmentations)()

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
            batch_size=self.cfg.training.train_batch_size,
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
            batch_size=self.cfg.training.valid_batch_size,
            num_workers=self.cfg.training.num_workers,
            shuffle=False,
            pin_memory=True,
        )

        return valid_loader

    def configure_optimizers(self):
        num_train_steps = len(self.train_dataloader()) * self.cfg.training.max_epochs
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg.training.lr)

        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=num_train_steps // 2, T_mult=1, eta_min=1e-7, last_epoch=-1
        )

        # lr_scheduler_method = hydra.utils.get_method(self.cfg.scheduler.method_name)
        # try:
        #     lr_scheduler = lr_scheduler_method(
        #         optimizer,
        #         num_training_steps=num_train_steps,
        #         **self.cfg.scheduler.params
        #     )
        # except:
        #     lr_scheduler = lr_scheduler_method(optimizer, **self.cfg.scheduler.params)

        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",  # or 'epoch'
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["label"]

        preds = self(images)
        loss = self.criterion(preds, labels)

        tensorboard_logs = {"train_loss": loss}

        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["label"]

        preds = self(images)
        loss = self.criterion(preds, labels)
        preds = torch.softmax(preds, dim=1)

        return {"preds": preds, "step_val_loss": loss}

    def validation_epoch_end(self, outputs):
        if len(outputs) == 0:
            return {"val_loss": 0}

        preds = np.vstack([x["preds"].cpu().detach().numpy() for x in outputs])

        labels = self.df_valid.growth_stage.values

        avg_loss = torch.stack([x["step_val_loss"] for x in outputs]).mean().item()

        preds = np.sum(preds * np.array(range(2, 8)), axis=-1)
        preds = np.clip(preds, 2, 7)
        rmse = np.sqrt(mean_squared_error(preds, labels))

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
