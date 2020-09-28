import gc
import glob
import logging

import hydra
import numpy as np
import omegaconf
import os
import pandas as pd
import pytorch_lightning as pl
import torch
import tqdm
import ttach
from sklearn import metrics
from torch.utils import data as torch_data

from src import dataset, lightning_models, utils

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def run_model(cfg: omegaconf.DictConfig) -> None:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    pl.seed_everything(cfg.general.seed)

    if cfg.testing.evaluate:
        test = pd.read_csv(cfg.data_mode.train_csv)
        test = test[test.label_quality == 2].reset_index(drop=True)
    elif cfg.testing.pseudolabels:
        test = pd.read_csv(cfg.data_mode.train_csv)
        test = test[test.label_quality == 1].reset_index(drop=True)
    else:
        test = pd.read_csv(cfg.testing.test_csv)
    test = utils.preprocess_df(test, data_dir=cfg.data_mode.data_dir)
    logger.info(f"Length of the test data: {len(test)}")

    device = torch.device("cuda")
    df_list = []
    pred_list = []

    for fold in cfg.testing.folds:

        if cfg.testing.evaluate:
            df_test = test[test.fold == fold].reset_index(drop=True)
        else:
            df_test = test

        checkpoints = glob.glob(
            os.path.join(
                cfg.general.logs_dir, f"model_{cfg.model.model_id}/fold_{fold}/*.ckpt"
            )
        )
        fold_predictions = np.zeros(
            (len(df_test), cfg.data_mode.num_classes, len(checkpoints))
        )

        for checkpoint_id, checkpoint_path in enumerate(checkpoints):
            model = lightning_models.LitWheatModel.load_from_checkpoint(
                checkpoint_path, hydra_cfg=cfg
            )
            model.eval().to(device)

            test_dataset = dataset.ZindiWheatDataset(
                images=df_test.path.values,
                labels=None,
                preprocess_function=model.preprocess,
                augmentations=None,
                input_shape=(cfg.model.input_size[0], cfg.model.input_size[1], 3),
                crop_method=cfg.model.crop_method,
            )

            if cfg.testing.tta:
                transforms = [
                    ttach.HorizontalFlip(),
                    # ttach.VerticalFlip(),
                    # ttach.Resize([(256, 256), (384, 384), (512, 512)]),
                    # ttach.Scale(scales=[1, 2]),
                    # ttach.Multiply(factors=[0.9, 1, 1.1]),
                ]
                if cfg.model.crop_method == "crop":
                    transforms.append(ttach.FiveCrops(crop_height=160, crop_width=512))
                transforms = ttach.Compose(transforms)
                model = ttach.ClassificationTTAWrapper(model, transforms)

            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)

            test_loader = torch_data.DataLoader(
                test_dataset,
                batch_size=cfg.training.batch_size,
                num_workers=cfg.general.num_workers,
                shuffle=False,
                pin_memory=True,
            )

            with torch.no_grad():
                tq = tqdm.tqdm(test_loader, total=len(test_loader))
                for idx, data in enumerate(tq):
                    images = data["image"]
                    images = images.to(device)

                    preds = model(images)
                    if not cfg.model.regression:
                        preds = torch.softmax(preds, dim=1)
                    preds = preds.cpu().detach().numpy()

                    fold_predictions[
                        idx
                        * cfg.training.batch_size : (idx + 1)
                        * cfg.training.batch_size,
                        :,
                        checkpoint_id,
                    ] = preds

        gc.collect()
        torch.cuda.empty_cache()
        fold_predictions = np.mean(fold_predictions, axis=-1)

        if cfg.testing.evaluate:
            df_list.append(df_test)

        pred_list.append(fold_predictions)

    if cfg.testing.evaluate:
        test = pd.concat(df_list)
        probs = np.vstack(pred_list)
        filename = "validation_probs.pkl"
    else:
        probs = np.stack(pred_list)
        probs = np.mean(probs, axis=0)
        filename = "pseudo_probs.pkl" if cfg.testing.pseudolabels else "test_probs.pkl"

    ensemble_probs = dict(zip(test.UID.values, probs))
    utils.save_in_file_fast(
        ensemble_probs,
        file_name=os.path.join(
            cfg.general.logs_dir, f"model_{cfg.model.model_id}/{filename}"
        ),
    )

    multipliers = np.array(cfg.data_mode.rmse_multipliers)
    if not cfg.model.regression:
        probs = np.sum(probs * multipliers, axis=-1)
    predictions = np.clip(probs, min(multipliers), max(multipliers))

    if cfg.testing.evaluate:
        rmse = np.sqrt(
            metrics.mean_squared_error(predictions, test.growth_stage.values)
        )
        logger.info(f"OOF VALIDATION SCORE: {rmse:.5f}")
    elif cfg.testing.pseudolabels:
        test["pred"] = predictions
        if len(cfg.testing.folds) == 1:
            save_path = os.path.join(
                cfg.general.logs_dir,
                f"model_{cfg.model.model_id}/bad_pseudo_fold_{cfg.testing.folds[0]}.csv",
            )
        else:
            save_path = os.path.join(
                cfg.general.logs_dir, f"model_{cfg.model.model_id}/bad_pseudo.csv"
            )
        logger.info(f"Saving pseudolabels to {save_path}")

        test[["UID", "growth_stage", "pred"]].to_csv(save_path, index=False)
    else:
        test["growth_stage"] = predictions
        save_path = os.path.join(
            cfg.general.logs_dir, f"model_{cfg.model.model_id}/test_preds.csv"
        )
        logger.info(f"Saving test predictions to {save_path}")

        test[["UID", "growth_stage"]].to_csv(save_path, index=False)


if __name__ == "__main__":
    run_model()
