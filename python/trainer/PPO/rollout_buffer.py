# python/trainer/PPO/rollout_buffer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch


@dataclass
class RolloutBatch:
    obs: torch.Tensor          # (B, C, H, W)
    actions: torch.Tensor      # (B,)
    logprobs: torch.Tensor     # (B,)
    advantages: torch.Tensor   # (B,)
    returns: torch.Tensor      # (B,)
    values: torch.Tensor       # (B,)


class RolloutBuffer:
    """
    Stores rollout data for PPO + computes GAE.
    """
    def __init__(self, n_steps: int, obs_shape: Tuple[int, int, int], device: torch.device):
        self.n_steps = n_steps
        self.obs_shape = obs_shape  # (C,H,W)
        self.device = device
        self.reset()

    def reset(self):
        self.ptr = 0
        self.full = False

        C, H, W = self.obs_shape
        self.obs = np.zeros((self.n_steps, C, H, W), dtype=np.float32)
        self.actions = np.zeros((self.n_steps,), dtype=np.int64)
        self.rewards = np.zeros((self.n_steps,), dtype=np.float32)
        self.dones = np.zeros((self.n_steps,), dtype=np.float32)
        self.values = np.zeros((self.n_steps,), dtype=np.float32)
        self.logprobs = np.zeros((self.n_steps,), dtype=np.float32)

    def add(self, obs, action, reward, done, value, logprob):
        # Prevent overflow
        if self.ptr >= self.n_steps:
            self.full = True
            return # ignore extra samples
        
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value
        self.logprobs[self.ptr] = logprob

        self.ptr += 1
        if self.ptr >= self.n_steps:
            self.full = True

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        """
        Produces advantages + returns.
        """
        adv = np.zeros((self.n_steps,), dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(self.n_steps)):
            nonterminal = 1.0 - self.dones[t]
            next_value = last_value if t == self.n_steps - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * nonterminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
            adv[t] = last_gae

        returns = adv + self.values
        return adv, returns

    def to_torch(self, advantages, returns) -> RolloutBatch:
        return RolloutBatch(
            obs=torch.from_numpy(self.obs).to(self.device),
            actions=torch.from_numpy(self.actions).to(self.device),
            logprobs=torch.from_numpy(self.logprobs).to(self.device),
            advantages=torch.from_numpy(advantages).to(self.device),
            returns=torch.from_numpy(returns).to(self.device),
            values=torch.from_numpy(self.values).to(self.device),
        )
    
