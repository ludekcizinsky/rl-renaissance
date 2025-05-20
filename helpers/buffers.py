from collections import defaultdict
import torch
import numpy as np


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

    def add_last_step(self, action, next_state, log_prob, value, reward):
        self._buf['actions'].append(action)
        self._buf['next_states'].append(next_state)
        self._buf['log_probs'].append(log_prob)
        self._buf['values'].append(value)
        self._buf['rewards'].append(reward)

    def aggregate_last_step(self):
        tensors_dict = self.to_tensors()
        out_dict = {}
        for key, seq in tensors_dict.items():
            if key in ["actions", "next_states"]:
                out_dict[key] = seq.mean(dim=0) # T x D -> D
            else:
                out_dict[key] = seq.mean() # T -> scalar

        agg_action = out_dict['actions']
        agg_next_states = out_dict['next_states']
        agg_log_probs = out_dict['log_probs']
        agg_values = out_dict['values']
        agg_rewards = out_dict['rewards'].detach().cpu().numpy()


        return agg_action, agg_next_states, agg_log_probs, agg_values, agg_rewards

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
                arr = np.array(seq)
                out[key] = torch.tensor(arr, dtype=torch.float32).detach()
        return out
    

    def clear(self):
        """Reset the buffer for a new trajectory."""
        self._buf.clear()
