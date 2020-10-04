"""Script for generating stacking on the first level predictions."""

import logging

import hydra
import lightgbm
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
        train = pd.read_csv(cfg.data_mode.train_csv, index_col="UID")
        train = train[["growth_stage", "fold"]]
        feature_columns = list(range(len(cfg.ensemble.model_ids)))

        predictions = utils.combine_dataframes(
            models_list=cfg.ensemble.model_ids,
            logs_dir=cfg.general.logs_dir,
            filename="valid_preds.csv",
            agg_func=None,
        )
        predictions.columns = feature_columns
        train = train.join(predictions, how="inner")

        lightgbm_params = {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 2,
            "learning_rate": 0.05,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.9,
            "bagging_freq": 5,
            "verbose": 1,
        }

        train["pred"] = -1
        multipliers = np.array(cfg.data_mode.rmse_multipliers)
        model_name = "_".join([str(x) for x in cfg.ensemble.model_ids])

        for fold in cfg.testing.folds:
            train_folds = [f for f in cfg.testing.folds if f != fold]

            x_train = train.loc[train.fold.isin(train_folds), feature_columns].values
            y_train = train.loc[train.fold.isin(train_folds), "growth_stage"].values

            x_test = train.loc[train.fold == fold, feature_columns].values
            y_test = train.loc[train.fold == fold, "growth_stage"].values

            train_data = lightgbm.Dataset(x_train, label=y_train)
            test_data = lightgbm.Dataset(x_test, label=y_test)

            gbm = lightgbm.train(
                lightgbm_params,
                train_data,
                valid_sets=test_data,
                num_boost_round=5000,
                early_stopping_rounds=100,
            )

            preds = gbm.predict(x_test)
            preds = np.clip(preds, min(multipliers), max(multipliers))
            train.loc[train.fold == fold, "pred"] = preds

            gbm.save_model(
                os.path.join(
                    cfg.general.logs_dir, f"{model_name}_stacking_fold_{fold}.txt"
                ),
                num_iteration=gbm.best_iteration,
            )

        rmse = np.sqrt(metrics.mean_squared_error(train.growth_stage, train.pred))
        logger.info(f"STACKING VALIDATION SCORE: {rmse:.5f}")
    else:
        test_predictions = utils.combine_dataframes(
            models_list=cfg.ensemble.model_ids,
            logs_dir=cfg.general.logs_dir,
            filename="test_preds.csv",
            agg_func=None,
        )
        feature_columns = list(range(len(cfg.ensemble.model_ids)))
        test_predictions.columns = feature_columns

        model_name = "_".join([str(x) for x in cfg.ensemble.model_ids])
        test_predictions["growth_stage"] = 0
        multipliers = np.array(cfg.data_mode.rmse_multipliers)

        for fold in cfg.testing.folds:
            gbm = lightgbm.Booster(
                model_file=os.path.join(
                    cfg.general.logs_dir, f"{model_name}_stacking_fold_{fold}.txt"
                )
            )

            preds = gbm.predict(test_predictions[feature_columns].values)
            preds = np.clip(preds, min(multipliers), max(multipliers))
            test_predictions["growth_stage"] += preds / len(cfg.testing.folds)

        save_path = os.path.join(
            cfg.general.logs_dir,
            f"{'_'.join([str(x) for x in cfg.ensemble.model_ids])}_ens.csv",
        )
        logger.info(f"Saving test predictions to {save_path}")
        test_predictions.reset_index()[["UID", "growth_stage"]].to_csv(save_path, index=False)


if __name__ == "__main__":
    make_ensemble()
