# python/trainer/main.py

from __future__ import annotations

from train_loop import TrainConfig, train


def main():
    cfg = TrainConfig()
    train(cfg)


if __name__ == "__main__":
    main()
