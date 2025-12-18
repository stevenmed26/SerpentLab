# python/trainer/main.py

from __future__ import annotations

import argparse
from trainer.common.registry import make_trainer
from trainer.common.config import load_config

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--run_id", default="default")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    trainer = make_trainer(
        algo=args.algo,
        cfg=cfg,
        run_id=args.run_id,
        device=args.device,
        seed=args.seed,
    )
    print("Beginning training with config:", cfg)
    trainer.train()


if __name__ == "__main__":
    main()
