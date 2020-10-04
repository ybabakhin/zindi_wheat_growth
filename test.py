"""Inference script for a single model.

Example:
        >>> python test.py model.model_id=1
"""

import gc
import glob
import logging

import hydra
import numpy as np
import omegaconf
import os
import pandas as pd
import torch
import tqdm
from sklearn import metrics
from torch.utils import data as torch_data

from src import dataset
from src import lightning_models
from src import tta
from src import utils

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def run_model(cfg: omegaconf.DictConfig) -> None:
    utils.setup_environment(seed=cfg.general.seed, gpu_list=cfg.general.gpu_list)

    if cfg.testing.mode == "valid":
        test = pd.read_csv(cfg.data_mode.train_csv)
        test = test[test.label_quality == 2].reset_index(drop=True)
    else:
        test = pd.read_csv(cfg.testing.test_csv)

    test = utils.preprocess_df(test, data_dir=cfg.data_mode.data_dir)
    logger.info(f"Length of the test data: {len(test)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_list = []
    pred_list = []

    for fold in cfg.testing.folds:
        if cfg.testing.mode == "valid":
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

            test_loader = torch_data.DataLoader(
                test_dataset,
                batch_size=cfg.training.batch_size,
                num_workers=cfg.general.num_workers,
                shuffle=False,
                pin_memory=True,
            )

            if cfg.testing.tta:
                model = tta.get_tta_model(
                    model,
                    crop_method=cfg.model.crop_method,
                    input_size=cfg.model.input_size,
                )

            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)

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

        # OOF predictions for validation and pseudolabels
        if cfg.testing.mode == "valid" or cfg.testing.mode == "pseudo":
            df_list.append(df_test)

        pred_list.append(fold_predictions)

    multipliers = np.array(cfg.data_mode.rmse_multipliers)

    if cfg.testing.mode == "valid":
        test = pd.concat(df_list)
        probs = np.vstack(pred_list)
        filename = "validation_probs.pkl"

    elif cfg.testing.mode == "pseudo":
        for fold, df_test, probs in zip(cfg.testing.folds, df_list, pred_list):
            predictions = np.argmax(probs, axis=1)
            predictions = [multipliers[x] for x in predictions]
            df_test["growth_stage"] = predictions
            save_path = os.path.join(
                cfg.general.logs_dir,
                f"model_{cfg.model.model_id}/pseudo_fold_{fold}.csv",
            )
            logger.info(f"Saving pseudolabels to {save_path}")
            df_test[["UID", "growth_stage"]].to_csv(save_path, index=False)
        return

    else:
        probs = np.stack(pred_list)
        probs = np.mean(probs, axis=0)
        filename = "test_probs.pkl"

    ensemble_probs = dict(zip(test.UID.values, probs))
    utils.save_in_file_fast(
        ensemble_probs,
        file_name=os.path.join(
            cfg.general.logs_dir, f"model_{cfg.model.model_id}/{filename}"
        ),
    )

    if not cfg.model.regression:
        probs = np.sum(probs * multipliers, axis=-1)
    predictions = np.clip(probs, min(multipliers), max(multipliers))

    if cfg.testing.mode == "valid":
        rmse = np.sqrt(
            metrics.mean_squared_error(predictions, test.growth_stage.values)
        )
        logger.info(f"OOF VALIDATION SCORE: {rmse:.5f}")

        test["pred"] = predictions
        save_path = os.path.join(
            cfg.general.logs_dir, f"model_{cfg.model.model_id}/valid_preds.csv"
        )
        logger.info(f"Saving validation predictions to {save_path}")
        test[["UID", "pred"]].to_csv(save_path, index=False)
    else:
        test["growth_stage"] = predictions
        save_path = os.path.join(
            cfg.general.logs_dir, f"model_{cfg.model.model_id}/test_preds.csv"
        )
        logger.info(f"Saving test predictions to {save_path}")
        test[["UID", "growth_stage"]].to_csv(save_path, index=False)


if __name__ == "__main__":
    run_model()
