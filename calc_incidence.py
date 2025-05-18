import torch


def evaluate_agent_incidence(cfg, actor_path, threshold, num_samples=100, logger=None):
    """
    Loads a pretrained agent and calculates the incidence rate.
    Incidence = percentage of samples with reward above the threshold.

    Args:
        cfg: Hydra config object.
        actor_path: Path to the saved actor/policy model.
        threshold: Reward threshold for incidence.
        num_samples: Number of samples to evaluate.
        logger: Optional logger.

    Returns:
        incidence_rate: Fraction of samples with reward > threshold.
        rewards: List of all sampled rewards.
    """
    # Setup environment and agent
    from helpers.ppo_agent import PPOAgent
    from helpers.env import KineticEnv
    from helpers.utils import reward_func, load_pkl
    from renaissance.kinetics.jacobian_solver import check_jacobian
    from functools import partial

    chk_jcbn = check_jacobian()
    chk_jcbn._load_ktmodels(cfg.paths.met_model_name, 'fdp1')
    chk_jcbn._load_ssprofile(cfg.paths.met_model_name, 'fdp1', cfg.constraints.ss_idx)
    names_km = load_pkl(cfg.paths.names_km)
    reward_fn = partial(reward_func, chk_jcbn, names_km, cfg.reward.eig_partition)
    env = KineticEnv(cfg, reward_fn)
    env.seed(cfg.seed)

    # Load agent and policy weights
    agent = PPOAgent(cfg, logger)
    agent.policy_net.load_state_dict(torch.load(actor_path, map_location="cpu"))
    agent.policy_net.eval()

    # Evaluate incidence
    count_above = 0
    rewards = []
    for _ in range(num_samples):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done and steps < cfg.training.max_steps_per_episode:
            with torch.no_grad():
                mean, std = agent.policy_net(state)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
            state, reward, done = env.step(action)
           
   
        if reward > threshold:
            count_above += 1
        print(f"Sample {_+1}/{num_samples}: Reward = {reward:.4f}")

    incidence_rate = count_above / num_samples
    print(f"Incidence rate (reward > {threshold}): {incidence_rate:.4f} ({count_above}/{num_samples})")
    return incidence_rate, rewards

if __name__ == "__main__":
    from omegaconf import OmegaConf
    import hydra
    from hydra import initialize, compose
    from renaissance.kinetics.jacobian_solver import check_jacobian

    from helpers.ppo_agent import PPOAgent
    from helpers.env import KineticEnv
    from helpers.utils import reward_func, load_pkl
    import torch

    # Hydra-style manual config loading in a notebook
    initialize(config_path="configs", version_base="1.1")
    cfg = compose(config_name="train.yaml")


    incidence, all_rewards = evaluate_agent_incidence(cfg, "output/best_actor.pth", threshold=0.5, num_samples=100)