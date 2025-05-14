import os

import wandb


def get_logger(cfg):
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    logger = wandb.init(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        config=cfg,
        tags=cfg.logger.tags,
        dir=cfg.paths.output_dir,
    )

    return logger