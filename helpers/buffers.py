from collections import defaultdict
import torch

class TrajectoryBuffer:
    def __init__(self):
        self._buf = defaultdict(list)

    def add(self, state, action, log_prob, value, reward, done):
        self._buf['states'].append(state)
        self._buf['actions'].append(action)
        self._buf['log_probs'].append(log_prob)
        self._buf['values'].append(value)
        self._buf['rewards'].append(reward)
        self._buf['dones'].append(done)

    def add_last_step(self, action, log_prob, value, reward):
        self._buf['actions'].append(action)
        self._buf['log_probs'].append(log_prob)
        self._buf['values'].append(value)
        self._buf['rewards'].append(reward)

    def aggregate_last_step(self):
        tensors_dict = self.to_tensors()
        out_dict = {}
        for key, seq in tensors_dict.items():
            if key in ["actions"]:
                out_dict[key] = seq.mean(dim=0) # T x D -> D
            else:
                out_dict[key] = seq.mean() # T -> scalar

        agg_action, agg_next_states = out_dict['actions'], out_dict['states']
        agg_log_probs, agg_values = out_dict['log_probs'], out_dict['values']
        agg_rewards = out_dict['rewards']

        return agg_action, agg_log_probs, agg_values, agg_next_states, agg_rewards

    def to_tensors(self):
        """Convert all lists into batched tensors."""
        out = {}
        for key, seq in self._buf.items():
            first = seq[0]
            if isinstance(first, torch.Tensor):
                # stack tensors (e.g. actions, values, log_probs)
                out[key] = torch.stack(seq).detach()
            else:
                # tensorify numeric types (e.g. rewards, dones, raw states)
                out[key] = torch.tensor(seq, dtype=torch.float32).detach()
        return out
    

    def clear(self):
        """Reset the buffer for a new trajectory."""
        self._buf.clear()
