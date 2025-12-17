# python/trainer/replay_buffer.py

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Transition:
    state: np.ndarray       # (H, W)
    action: int
    reward: float
    next_state: np.ndarray  # (H, W)
    done: bool
    gamma_n: float = 0.99


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000, alpha: float = 0.6, eps: float = 1e-6):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.eps = float(eps)
        
        self.states = [None] * self.capacity
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = [None] * self.capacity
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.gammas = np.ones(self.capacity, dtype=np.float32)

        self.priorities = np.zeros(self.capacity, dtype=np.float32)

        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

    def push(self, t: Transition):
        i = self.pos

        self.states[i] = t.state
        self.actions[i] = t.action
        self.rewards[i] = t.reward
        self.next_states[i] = t.next_state
        self.dones[i] = float(t.done)
        self.gammas[i] = float(t.gamma_n)

        self.priorities[i] = self.max_priority

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self) -> int:
        return self.size

    def sample(self, batch_size: int, beta: float) -> Tuple[np.ndarray, ...]:
        assert self.size > 0
        batch_size = int(batch_size)

        prios = self.priorities[: self.size]
        if prios.sum() == 0:
            prios = np.ones_like(prios)
        
        probs = prios ** self.alpha
        probs /= probs.sum()

        idxs = np.random.choice(self.size, size=batch_size, replace=False, p=probs)

        weights = (self.size * probs[idxs]) ** (-beta)
        weights /= weights.max()

        states = np.stack([self.states[i] for i in idxs], axis=0)
        next_states = np.stack([self.next_states[i] for i in idxs], axis=0)

        return (
            states,                      # (B, H, W)
            self.actions[idxs].copy(),         # (B,)
            self.rewards[idxs].copy(),         # (B,)
            next_states,                # (B, H, W)
            self.dones[idxs].copy(),           # (B,)
            self.gammas[idxs].copy(),          # (B,)
            idxs.astype(np.int64),                       # (B,)
            weights.astype(np.float32), # (B,)
        )
    
    def update_priorities(self, idxs: np.ndarray, priorities: np.ndarray):
        priorities = np.asarray(priorities, dtype=np.float32)
        idxs = np.asarray(idxs, dtype=np.int64)

        priorities = np.maximum(priorities, self.eps)
        self.priorities[idxs] = priorities
        self.max_priority = float(max(self.max_priority, priorities.max()))
