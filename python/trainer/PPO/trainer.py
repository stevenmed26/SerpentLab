# python/trainer/PPO/trainer.py
from __future__ import annotations
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.distributions import Categorical

from trainer.PPO.model import SnakeActorCritic
from trainer.common.obs import one_hot_obs
from trainer.PPO.train_loop import train_ppo, PPOConfig


class BaseTrainer:
    def train(self): ...
    def save(self, path: str): ...
    def load(self, path: str): ...
    def act(self, obs): ...

class PPOTrainer:
    def __init__(self, *, config: Any, device: str, seed: int, run_id: str):
        self.config = config
        self.device_str = device
        self.seed = seed
        self.run_id = run_id

        self.device = torch.device(device)
        self.model: Optional[SnakeActorCritic] = None

    def build_model(self, height: int, width: int, num_actions: int = 4):
        self.model = SnakeActorCritic(height=height, width=width, num_actions=num_actions).to(self.device)
        self.model.eval()

    def act(self, obs_grid: np.ndarray) -> int:
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        x = one_hot_obs(obs_grid, num_channels=4)  # (C, H, W)
        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)  # (1, C, H, W)
        with torch.no_grad():
            logits, _value = self.model(xt)
            m = Categorical(logits=logits)
            action = m.sample().item()
        return int(action.item())

    def train(self):
        return train_ppo(self.train_cfg, device=self.device, seed=self.seed)