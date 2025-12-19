# python/trainer/DQN/trainer.py
from __future__ import annotations
from dataclasses import asdict

import os
import torch

from trainer.DQN.train_loop import DQNConfig, train
# from trainer.common.checkpoint import save_checkpoint, load_checkpoint

class BaseTrainer:
    def train(self): ...
    def save(self, path: str): ...
    def load(self, path: str): ...
    def act(self, obs): ...

class DQNTrainer:
    def __init__(self, *, config, run_id: str, device: str, seed: int):
        self.run_id = run_id
        self.device = device
        self.seed = seed

        if isinstance(config, dict):
            self.train_cfg = DQNConfig(**config)
        else:
            self.train_cfg = config

    def train(self):
        return train(self.train_cfg, device=self.device, seed=self.seed)
    