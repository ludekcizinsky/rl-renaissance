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
        self._reset_generator.manual_seed(cfg.seed)

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

class BatchKineticEnv:
    def __init__(self, cfg, reward_fn, batch_size):
        self.param_dim = cfg.env.p_size
        self.min_val = cfg.constraints.min_km
        self.max_val = cfg.constraints.max_km
        self.reward_fn = reward_fn
        self.max_steps = cfg.training.max_steps_per_episode
        self.device = cfg.device
        self.batch_size = batch_size

        self._reset_generator = torch.Generator(device=self.device)
        self._reset_generator.manual_seed(cfg.seed)

        self.state = None
        self.step_count = 0

    def reset(self) -> torch.Tensor:
        self.step_count = 0
        self.state = torch.rand(
            (self.batch_size, self.param_dim),
            generator=self._reset_generator,
            device=self.device
        ) * (self.max_val - self.min_val) + self.min_val
        return self.state.clone()

    def step(self, action: torch.Tensor):
        """
        Args:
            action: Tensor of shape (batch_size, param_dim)
        Returns:
            next_state (Tensor), reward (Tensor), done (Tensor)
        """
        action = action.to(self.device)
        self.state = (self.state + action).clamp(
            min=self.min_val, max=self.max_val
        )
        rewards = self.reward_fn(self.state, self.step_count / self.max_steps)
        self.step_count += 1
        done = torch.full((self.batch_size,), self.step_count >= self.max_steps, dtype=torch.bool, device=self.device)
        return self.state.clone(), rewards, done

    def render(self):
        vec = self.state.cpu().numpy()
        print(f"Step {self.step_count:3d} | params = {vec}")

    def seed(self, seed: int):
        torch.manual_seed(seed)
