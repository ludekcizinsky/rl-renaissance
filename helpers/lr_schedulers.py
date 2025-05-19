from torch.optim.lr_scheduler import LinearLR, ConstantLR, CosineAnnealingLR


def get_lr_scheduler(cfg, optimizer):
    if cfg.lr_scheduler.name == "constant":
        return get_constant_lr(optimizer, cfg)
    elif cfg.lr_scheduler.name == "linear_decay":
        return get_linear_decay(optimizer, cfg)
    elif cfg.lr_scheduler.name == "cosine":
        return get_cosine_annealing(optimizer, cfg)
    else:
        raise ValueError(f"Unknown LR scheduler: {cfg.lr_scheduler.name}")

def _get_total_updates(cfg):
    n_episodes = cfg.training.num_episodes
    n_epochs_per_episode = cfg.training.num_epochs
    n_updates_per_epoch = cfg.training.max_steps_per_episode / cfg.training.batch_size
    total_updates = n_episodes * n_epochs_per_episode * n_updates_per_epoch
    return total_updates

def get_cosine_annealing(optimizer, cfg):
    total_updates = _get_total_updates(cfg)
    cos_annealing = CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=cfg.lr_scheduler.eta)
    return cos_annealing

def get_constant_lr(optimizer, cfg):
    return ConstantLR(optimizer, factor=1.0)

def get_linear_decay(optimizer, cfg):
    total_updates = _get_total_updates(cfg)
    decay = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_updates)
    return decay
