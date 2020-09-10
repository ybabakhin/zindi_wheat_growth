import glob
import os
from pytorch_lightning import seed_everything
import pandas as pd
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
from src.utils import preprocess_df

import hydra
from omegaconf import DictConfig
import gc
from src.dataset import ZindiWheatDataset
from src.lightning_models import LitWheatModel
from sklearn.metrics import mean_squared_error
from src.utils import save_in_file_fast


@hydra.main(config_path="conf", config_name="config")
def run_model(cfg: DictConfig):

    if cfg.testing.evaluate:
        test = pd.read_csv(cfg.training.train_csv)
        test = preprocess_df(test, data_dir=cfg.training.data_dir)

        test = test[test.label_quality == 2].reset_index(drop=True)
    elif cfg.testing.pseudolabels:
        pass
    else:
        test = pd.read_csv(cfg.testing.test_csv)
        test = preprocess_df(test, data_dir=cfg.training.data_dir)

    seed_everything(cfg.training.seed)
    device = torch.device("cuda")
    df_list = []
    pred_list = []

    for fold in cfg.testing.folds:

        if cfg.testing.evaluate:
            df_test = test[test.fold == fold].reset_index(drop=True)
        else:
            df_test = test

        checkpoints = glob.glob(
            f"/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_{cfg.training.model_id}/fold_{fold}/*.ckpt"
        )
        fold_predictions = np.zeros(
            (len(df_test), cfg.training.num_classes, len(checkpoints))
        )

        for checkpoint_id, checkpoint_path in enumerate(checkpoints):
            checkpoint = torch.load(checkpoint_path)
            model = LitWheatModel(cfg)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval().to(device)

            test_dataset = ZindiWheatDataset(
                images=df_test.path.values,
                labels=None,
                preprocess_function=model.preprocess,
                augmentations=None,
                input_shape=(cfg.training.input_size, cfg.training.input_size, 3),
                crop_function="resize",
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg.training.valid_batch_size,
                num_workers=cfg.training.num_workers,
                shuffle=False,
                pin_memory=True,
            )

            with torch.no_grad():
                tq = tqdm(test_loader, total=len(test_loader))
                for idx, data in enumerate(tq):
                    images = data["image"]
                    images = images.to(device)

                    preds = model(images)
                    preds = torch.softmax(preds, dim=1).cpu().detach().numpy()

                    fold_predictions[
                        idx
                        * cfg.training.valid_batch_size : (idx + 1)
                        * cfg.training.valid_batch_size,
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
        filename = "test_probs.pkl"

    ensemble_probs = dict(zip(test.UID.values, probs))
    save_in_file_fast(
        ensemble_probs,
        file_name=os.path.join(
            f"/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_{cfg.training.model_id}/",
            filename,
        ),
    )
    predictions = np.sum(probs * np.array([2, 3, 4, 5, 7]), axis=-1)
    predictions = np.clip(predictions, 2, 7)

    if cfg.testing.evaluate:
        rmse = np.sqrt(mean_squared_error(predictions, test.growth_stage.values))
        print(f"OOF VALIDATION SCORE: {rmse:.5f}")
    else:
        test["growth_stage"] = predictions
        test[["UID", "growth_stage"]].to_csv(
            os.path.join(
                f"/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lightning_logs/model_{cfg.training.model_id}/",
                "test_preds.csv",
            ),
            index=False,
        )


if __name__ == "__main__":
    run_model()
