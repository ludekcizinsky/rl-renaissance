class SweepRegistry:
    """Registry for parameters to include in hyperparameter sweeps."""
    
    registry = {}
    
    @classmethod
    def register(cls, param_path, range_type, **range_args):
        """
        Register a parameter for sweeping.
        
        Args:
            param_path: The dot-notation path to the parameter in the config
            range_type: Type of range ('uniform', 'log_uniform_values', 'values', etc.)
            range_args: Arguments specific to the range type
        """
        cls.registry[param_path] = {"type": range_type, **range_args}
        
    @classmethod
    def get_sweep_config(cls):
        """Generate a W&B sweep config from the registered parameters."""
        params = {}
        for param_path, range_spec in cls.registry.items():
            range_spec_copy = range_spec.copy()  # Make a copy to avoid modifying the original
            param_type = range_spec_copy.pop("type")
            
            if param_type == "uniform":
                params[param_path] = {"min": range_spec_copy["min"], "max": range_spec_copy["max"]}
            elif param_type == "log_uniform_values":
                params[param_path] = {"distribution": "log_uniform_values", 
                                     "min": range_spec_copy["min"], 
                                     "max": range_spec_copy["max"]}
            elif param_type == "values":
                params[param_path] = {"values": range_spec_copy["values"]}
                
        return {
            'method': 'random',
            'metric': {'name': 'mean_reward', 'goal': 'maximize'},
            'parameters': params
        }