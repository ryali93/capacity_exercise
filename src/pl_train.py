import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import torch, numpy as np, random

from config import Cfg
from pl_data import MSDataModule
from pl_module import LitSeg

def train(cfg_path="config.yaml"):
    cfg = Cfg.load(cfg_path)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    dm = MSDataModule(cfg)
    model = LitSeg(in_ch=len(dm.wanted), cfg=cfg, channels=dm.wanted)

    # logger
    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(project=cfg.wandb.project, entity=None,
                             name=cfg.wandb.name, mode=cfg.wandb.mode, tags=cfg.wandb.tags,
                             log_model=False)

    # callbacks
    ckpt_dir = os.path.join(cfg.logging.out_dir, cfg.wandb.name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = ModelCheckpoint(
        dirpath=ckpt_dir, filename="best",
        monitor="val/f1", mode="max",
        save_weights_only=True, save_top_k=1
    )
    early = EarlyStopping(monitor="val/f1", mode="max",
                          patience=cfg.training.early_stopping_patience)

    tr = pl.Trainer(
        max_epochs=cfg.training.epochs,
        callbacks=[ckpt, early],
        logger=logger,
        default_root_dir=ckpt_dir,
        enable_checkpointing=True,
    )

    tr.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    logger.experiment.finish()

    # save last
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "last.pth"))
