# python/trainer/main.py

from __future__ import annotations

from python.trainer.DQN.train_loop import TrainConfig, train


def main():
    cfg = TrainConfig()
    print("Beginning training with config:", cfg)
    train(cfg)


if __name__ == "__main__":
    main()
