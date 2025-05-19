import os
from omegaconf import OmegaConf, DictConfig
import wandb


def get_wandb_run(cfg: DictConfig):
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    run = wandb.init(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=cfg.logger.tags,
        dir=cfg.paths.output_dir,
    )

    run.define_metric(name="ppo/*", step_metric="global_step")
    run.define_metric(name="optim/*", step_metric="global_step")
    run.define_metric(name="reward/*", step_metric="episode")
    run.define_metric(name="episode/*", step_metric="env_step")
    return run