import torch

class KineticEnv:
    def __init__(self, cfg, reward_fn):
        """
        Environment for kinetic model parameters refinement.
        """

        self.param_dim = cfg.env.p_size
        self.min_val = cfg.constraints.min_km
        self.max_val = cfg.constraints.max_km
        self.reward_fn = reward_fn
        self.max_steps = cfg.training.max_steps_per_episode
        self.device = cfg.device

        self._reset_generator = torch.Generator(device=self.device)
        self._reset_generator.manual_seed(self.reset_seed)

        self.state = None
        self.step_count = 0

    def reset(self) -> torch.Tensor:
        """Sample the same deterministic initial params on every reset."""
        self.step_count = 0
        # use the fixed generator to sample initial state
        self.state = torch.rand(
            (self.param_dim,),
            generator=self._reset_generator,
            device=self.device
        ) * (self.max_val - self.min_val) + self.min_val
        return self.state.clone()

    def step(self, action: torch.Tensor):
        """
        Apply delta-action, clamp to integer bounds, and compute reward.

        Args:
            action: Tensor of shape (param_dim,) to add to state
        Returns:
            next_state (Tensor), reward (float), done (bool)
        """
        action = action.to(self.device)
        # update and clamp
        self.state = (self.state + action).clamp(
            min=self.min_val, max=self.max_val
        )

        # compute black-box reward
        r = self.reward_fn(self.state)
        reward = float(r) if isinstance(r, torch.Tensor) else r

        # increment and check termination
        self.step_count += 1
        done = (self.step_count >= self.max_steps)

        return self.state.clone(), reward, done

    def render(self):
        """Print current state of parameters."""
        vec = self.state.cpu().numpy()
        print(f"Step {self.step_count:3d} | params = {vec}")

    def seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)


# Example usage:
if __name__ == "__main__":
    # Hidden target in [0,10]^4
    target = torch.tensor([2.5, 7.3, 1.1, 4.8])
    def blackbox_reward(x):
        return -torch.norm(x - target).item()

    # reset_seed ensures same initial state every reset
    env = KineticEnv(
        param_dim=4,
        min_val=0,
        max_val=10,
        reward_fn=blackbox_reward,
        max_steps=50,
        reset_seed=12345,
    )
    env.seed(999)

    # Two resets yield identical states:
    s1 = env.reset()
    s2 = env.reset()
    print("Deterministic resets equal?", torch.allclose(s1, s2))

    total_reward = 0.0
    done = False
    state = s1
    while not done:
        # simple greedy step toward target
        direction = (target - state)
        action = direction / (direction.norm() + 1e-8) * 0.2
        state, r, done, _ = env.step(action)
        total_reward += r

    print("Episode finished. Total reward:", total_reward)