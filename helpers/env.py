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
        self.action_scale = cfg.env.action_scale

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
        action = self.action_scale * action
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

class KineticEnvStateless:
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

        self.state = None
        self.step_count = 0
        self.std = cfg.env.p0_init_std

    def reset(self) -> torch.Tensor:
        """Sample the same deterministic initial params on every reset."""
        # use the fixed generator to sample initial state
        self.state = self.std * torch.rand(
            (self.param_dim,),
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
        action = (action).clamp(
            min=self.min_val, max=self.max_val
        )

        # compute black-box reward
        r = self.reward_fn(action)
        reward = float(r) if isinstance(r, torch.Tensor) else r

        self.state = self.std * torch.rand(
            (self.param_dim,),
            device=self.device
        ) * (self.max_val - self.min_val) + self.min_val

        return self.state.clone(), reward, None

    def render(self):
        """Print current state of parameters."""
        return # TODO: keep the same function signature as the other env

    def seed(self, seed: int):
        """Set random seed for reproducibility."""
        return #TODO: keep the same function signature as the other env
