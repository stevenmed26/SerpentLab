# python/trainer/main.py

from __future__ import annotations

from train_loop import TrainConfig, train


def main():
    cfg = TrainConfig(
        address="localhost:50051",
        width=10,
        height=10,
        with_walls=True,
        num_episodes=200,            # start small while debugging
        max_steps_per_episode=300,
        batch_size=64,
        learning_rate=1e-3,
    )
    train(cfg)


if __name__ == "__main__":
    main()
