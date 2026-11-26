import torch

class RolloutBuffer:
    def __init__(self, capacity, device):
        self.device = device
        self.capacity = capacity
        self.clear()

    def add(self, num, img, act, lp, val, rw, done):
        self.numerics.append(num)
        self.images.append(img)
        self.actions.append(act)
        self.logps.append(lp)
        self.values.append(val)
        self.rewards.append(rw)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_val, gamma=0.99, lam=0.95):
        """
        Compute advantages and returns using GAE.
        Args:
            last_val (float): The value estimate for the last (bootstrap) state.
            gamma (float): Discount factor.
            lam (float): GAE smoothing parameter (lambda).
        """
        # Append the last value for bootstrap
        vals = self.values + [last_val]
        advs = []
        gae = 0.0
        # Compute GAE advantage
        for i in reversed(range(len(self.rewards))):
            mask = 1.0 - float(self.dones[i])
            delta = self.rewards[i] + gamma * vals[i+1] * mask - vals[i]
            gae = delta + gamma * lam * mask * gae
            advs.insert(0, gae)
        # Store tensors
        self.advantages = torch.tensor(advs, device=self.device)
        self.returns = self.advantages + torch.tensor(self.values, device=self.device)
        # Concatenate trajectory data
        self.numerics = torch.cat(self.numerics, dim=0)
        self.images   = torch.cat(self.images,   dim=0)
        self.actions  = torch.cat(self.actions,  dim=0)
        self.logps    = torch.cat(self.logps,    dim=0)
        self.values   = torch.tensor(self.values, device=self.device)
        self.dones    = torch.tensor(self.dones,   device=self.device)

    def clear(self):
        """
        Reset all storage lists for a new rollout.
        """
        self.numerics = []
        self.images   = []
        self.actions  = []
        self.logps    = []
        self.values   = []
        self.rewards  = []
        self.dones    = []
