import os
from omegaconf import OmegaConf, DictConfig
import wandb


def get_logger(cfg: DictConfig):
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    logger = wandb.init(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=cfg.logger.tags,
        dir=cfg.paths.output_dir,
    )

    return logger