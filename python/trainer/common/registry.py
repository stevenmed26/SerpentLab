#python/trainer/common/registry.py
from __future__ import annotations
from typing import Any

def make_trainer(*, algo: str, cfg: Any, device: str, seed: int, run_id: str):
    algo = algo.lower()
    if algo == "dqn":
        from python.trainer.DQN.train_loop import DQNTrainer
        return DQNTrainer(cfg=cfg, device=device, seed=seed, run_id=run_id)
    elif algo == "ppo":
        from python.trainer.PPO.train_loop import PPOTrainer
        return PPOTrainer(cfg=cfg, device=device, seed=seed, run_id=run_id)
    raise ValueError(f"Unknown algorithm: {algo}")