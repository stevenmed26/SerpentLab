#python/trainer/common/registry.py
from __future__ import annotations
from typing import Any

def make_trainer(*, algo: str, cfg: Any, device: str, seed: int, run_id: str):
    algo = algo.lower()
    if algo == "dqn":
        from trainer.DQN.trainer import DQNTrainer
        return DQNTrainer(config=cfg, device=device, seed=seed, run_id=run_id)
    elif algo == "ppo":
        from trainer.PPO.trainer import PPOTrainer
        return PPOTrainer(config=cfg, device=device, seed=seed, run_id=run_id)
    raise ValueError(f"Unknown algorithm: {algo}")