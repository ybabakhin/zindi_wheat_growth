"""Script for generating blend of the input models."""

import logging

import hydra
import numpy as np
import omegaconf
import os
import pandas as pd
from sklearn import metrics

from src import utils

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def make_ensemble(cfg: omegaconf.DictConfig) -> None:
    utils.setup_environment(seed=cfg.general.seed, gpu_list=cfg.general.gpu_list)

    if cfg.testing.mode == "valid":
        train = pd.read_csv(cfg.data_mode.train_csv)

        predictions = utils.combine_dataframes(
            models_list=cfg.ensemble.model_ids,
            logs_dir=cfg.general.logs_dir,
            filename="valid_preds.csv",
            output_colname="pred",
        )
        predictions = predictions.merge(train)

        rmse = np.sqrt(
            metrics.mean_squared_error(predictions.growth_stage, predictions.pred)
        )
        logger.info(f"OOF ENSEMBLE VALIDATION SCORE: {rmse:.5f}")
    elif cfg.testing.mode == "pseudo":
        for fold in cfg.testing.folds:
            test_predictions = utils.combine_dataframes(
                models_list=cfg.ensemble.model_ids,
                logs_dir=cfg.general.logs_dir,
                filename=f"pseudo_fold_{fold}.csv",
                agg_func="mode",
            )

            save_path = os.path.join(
                cfg.general.logs_dir,
                f"{'_'.join([str(x) for x in cfg.ensemble.model_ids])}_pseudo_fold_{fold}.csv",
            )
            logger.info(f"Saving pseudo predictions to {save_path}")
            test_predictions[["UID", "growth_stage"]].to_csv(save_path, index=False)
    else:
        test_predictions = utils.combine_dataframes(
            models_list=cfg.ensemble.model_ids,
            logs_dir=cfg.general.logs_dir,
            filename="test_preds.csv",
        )

        save_path = os.path.join(
            cfg.general.logs_dir,
            f"{'_'.join([str(x) for x in cfg.ensemble.model_ids])}_ens.csv",
        )
        logger.info(f"Saving test predictions to {save_path}")
        test_predictions[["UID", "growth_stage"]].to_csv(save_path, index=False)


if __name__ == "__main__":
    make_ensemble()
