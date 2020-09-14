import hydra
import os
import torch
from src.lightning_models import LitWheatModel
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from omegaconf import DictConfig
import glob


@hydra.main(config_path="conf", config_name="config")
def run_model(cfg: DictConfig):
    os.environ["HYDRA_FULL_ERROR"] = "1"
    seed_everything(cfg.training.seed)

    earlystopping_callback = hydra.utils.instantiate(cfg.callbacks.early_stopping)
    checkpoint_callback = hydra.utils.instantiate(cfg.callbacks.model_checkpoint)
    tb_logger = hydra.utils.instantiate(cfg.callbacks.tensorboard)
    lr_logger = hydra.utils.instantiate(cfg.callbacks.lr_logger)

    if cfg.training.pretrain_dir != "":
        pretrain_path = glob.glob(os.path.join(cfg.training.pretrain_dir, "*.ckpt"))[0]
        model = LitWheatModel.load_from_checkpoint(pretrain_path, hydra_cfg=cfg)

        # Number of classes in bad labels does not equal to the number of classes in good labels
        fc_layer_name = (
            "_fc" if cfg.training.architecture_name.startswith("efficientnet") else "_classifier"
        )
        if getattr(model.model, fc_layer_name).out_features != cfg.data_mode.num_classes:
            fc = torch.nn.Linear(
                getattr(model.model, fc_layer_name).in_features, cfg.data_mode.num_classes
            )
            setattr(model.model, fc_layer_name, fc)
    else:
        model = LitWheatModel(hydra_cfg=cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        min_epochs=cfg.training.max_epochs,
        logger=[tb_logger],
        early_stop_callback=earlystopping_callback,
        checkpoint_callback=checkpoint_callback,
        callbacks=[lr_logger],
        gradient_clip_val=0.5,
        gpus=cfg.training.gpu_list,
        fast_dev_run=False,
        distributed_backend="dp",
        precision=32,
        weights_summary=None,
        progress_bar_refresh_rate=5,
        deterministic=True,
    )

    # model.setup()
    # # Run learning rate finder
    # lr_finder = trainer.lr_find(model)
    # fig = lr_finder.plot(suggest=True)
    # fig.savefig("/data/ybabakhin/data/zindi_wheat/zindi_wheat_growth/lrfinder.png")

    trainer.fit(model)


if __name__ == "__main__":
    run_model()
