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
