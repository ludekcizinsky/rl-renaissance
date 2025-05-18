from helpers.sweep_registry import SweepRegistry

# Register parameters with their sweep ranges - note the change to log_uniform_values
SweepRegistry.register("method.actor_lr", "log_uniform_values", min=1e-5, max=1e-3)
SweepRegistry.register("method.critic_lr", "log_uniform_values", min=1e-5, max=1e-3)
SweepRegistry.register("method.discount_factor", "log_uniform_values", min=0.9, max=0.999)
SweepRegistry.register("method.gae_lambda", "log_uniform_values", min=0.8, max=1.0)
SweepRegistry.register("method.value_loss_weight", "values", values=[0.5, 1.0, 2.0])
SweepRegistry.register("method.entropy_loss_weight", "log_uniform_values", min=0.01, max=0.1)
SweepRegistry.register("method.parameter_dim", "values", values=[1, 10, 100, 250, 500, 1000])
# Add as many parameters as needed