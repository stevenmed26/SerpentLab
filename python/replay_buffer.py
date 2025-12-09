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


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Transition):
        self.buffer.append(transition)

    def __len__(self) -> int:
        return len(self.buffer)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for i in idx:
            t = self.buffer[i]
            states.append(t.state)
            actions.append(t.action)
            rewards.append(t.reward)
            next_states.append(t.next_state)
            dones.append(t.done)

        return (
            np.stack(states, axis=0),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states, axis=0),
            np.array(dones, dtype=np.float32),
        )
