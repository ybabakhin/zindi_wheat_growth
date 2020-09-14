import gc
import hydra
import glob
import os
from pytorch_lightning import seed_everything
import pandas as pd
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
from src.utils import preprocess_df
from omegaconf import DictConfig
from src.dataset import ZindiWheatDataset
from src.lightning_models import LitWheatModel
from sklearn.metrics import mean_squared_error
from src.utils import save_in_file_fast
import ttach as tta


@hydra.main(config_path="conf", config_name="config")
def run_model(cfg: DictConfig):
    os.environ["HYDRA_FULL_ERROR"] = "1"
    seed_everything(cfg.training.seed)

    if cfg.testing.evaluate:
        test = pd.read_csv(cfg.data_mode.train_csv)
        test = preprocess_df(test, data_dir=cfg.data_mode.data_dir)
        test = test[test.label_quality == 2].reset_index(drop=True)
    elif cfg.testing.pseudolabels:
        pass
    else:
        test = pd.read_csv(cfg.testing.test_csv)
        test = preprocess_df(test, data_dir=cfg.data_mode.data_dir)

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
                cfg.training.logs_dir, f"model_{cfg.training.model_id}/fold_{fold}/*.ckpt"
            )
        )
        fold_predictions = np.zeros((len(df_test), cfg.data_mode.num_classes, len(checkpoints)))

        for checkpoint_id, checkpoint_path in enumerate(checkpoints):
            model = LitWheatModel.load_from_checkpoint(checkpoint_path, hydra_cfg=cfg)
            model.eval().to(device)

            test_dataset = ZindiWheatDataset(
                images=df_test.path.values,
                labels=None,
                preprocess_function=model.preprocess,
                augmentations=None,
                input_shape=(cfg.training.input_size, cfg.training.input_size, 3),
                crop_function="resize",
            )

            if cfg.testing.tta:
                transforms = tta.Compose(
                    [
                        tta.HorizontalFlip(),
                        # tta.Resize([(256, 256), (384, 384), (512, 512)]),
                        # tta.Scale(scales=[1, 2]),
                        # tta.Multiply(factors=[0.9, 1, 1.1]),
                    ]
                )
                model = tta.ClassificationTTAWrapper(model, transforms)

            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)

            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg.training.batch_size,
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
                    if not cfg.training.regression:
                        preds = torch.softmax(preds, dim=1)
                    preds = preds.cpu().detach().numpy()

                    fold_predictions[
                        idx * cfg.training.batch_size : (idx + 1) * cfg.training.batch_size,
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
            cfg.training.logs_dir, f"model_{cfg.training.model_id}/{filename}"
        ),
    )

    multipliers = np.array(cfg.data_mode.rmse_multipliers)
    if not cfg.training.regression:
        probs = np.sum(probs * multipliers, axis=-1)
    predictions = np.clip(probs, min(multipliers), max(multipliers))

    if cfg.testing.evaluate:
        rmse = np.sqrt(mean_squared_error(predictions, test.growth_stage.values))
        print(f"OOF VALIDATION SCORE: {rmse:.5f}")
    else:
        test["growth_stage"] = predictions
        test[["UID", "growth_stage"]].to_csv(
            os.path.join(cfg.training.logs_dir, f"model_{cfg.training.model_id}/test_preds.csv"),
            index=False,
        )


if __name__ == "__main__":
    run_model()
