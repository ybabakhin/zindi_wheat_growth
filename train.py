import os
from pytorch_lightning.callbacks import LearningRateLogger
from src.lightning_models import LitWheatModel
from argparse import Namespace
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import hydra
from omegaconf import DictConfig
import torch
from src.utils import save_in_file_fast, load_from_file_fast
import glob


@hydra.main(config_path="conf", config_name="config")
def run_model(cfg: DictConfig):
    print(cfg.pretty())
    print("Working directory : {}".format(os.getcwd()))

    earlystopping_callback = hydra.utils.instantiate(cfg.callbacks.early_stopping)
    checkpoint_callback = hydra.utils.instantiate(cfg.callbacks.model_checkpoint)
    tb_logger = hydra.utils.instantiate(cfg.callbacks.tensorboard)
    lr_logger = LearningRateLogger()

    seed_everything(cfg.training.seed)

    if cfg.training.pretrain_path == "":
        model = LitWheatModel(hydra_cfg=cfg)
    else:
        pass
        # checkpoint = load_from_file_fast(cfg.training.pretrain_path)
        # hparams = Namespace(**checkpoint["hparams"])
        # model = LitTweetModel(hparams, hydra_cfg=cfg)
        # model.load_state_dict(checkpoint["state_dict"])

    gpu_list = [int(x) for x in cfg.training.gpu_list.split(",") if x != ""]

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        min_epochs=cfg.training.max_epochs,
        logger=[tb_logger],
        early_stop_callback=earlystopping_callback,
        checkpoint_callback=checkpoint_callback,
        callbacks=[lr_logger],
        num_sanity_val_steps=0,
        gradient_clip_val=0.5,
        gpus=gpu_list,
        progress_bar_refresh_rate=1,
        fast_dev_run=False,
        # train_percent_check=0.1,
        # distributed_backend="dp",
        row_log_interval=100,
        accumulate_grad_batches=1,
        # amp_level="O1",
        weights_summary=None,
    )

    trainer.fit(model)

    # model_path = glob.glob(
    #     os.path.join(
    #         cfg.training.logs_dir,
    #         f"model_{cfg.training.model_id}",
    #         f"fold_{cfg.training.fold}",
    #         "*.ckpt",
    #     )
    # )[0]
    #
    # checkpoint = torch.load(model_path, map_location="cuda:0")
    # cc = {k: v for k, v in checkpoint.items() if k in ["state_dict", "hparams"]}
    #
    # checkpoint_path = os.path.join(
    #     cfg.training.logs_dir,
    #     f"model_{cfg.training.model_id}",
    #     f"fold_{cfg.training.fold}",
    #     "best.pkl",
    # )
    #
    # save_in_file_fast(cc, checkpoint_path)

    # if not cfg.training.pseudolabels:
    # os.remove(model_path)


if __name__ == "__main__":
    run_model()
