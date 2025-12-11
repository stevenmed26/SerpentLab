import os
import numpy as np
import torch

from train_loop import TrainConfig, SnakeDQN, SnakeRemoteEnv  # adjust imports if paths differ


def eval_checkpoint(
    checkpoint_path: str,
    num_episodes: int = 50,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_dict = ckpt.get("config", {})
    width = cfg_dict.get("width", 10)
    height = cfg_dict.get("height", 10)

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Checkpoint width={width}, height={height}")

    # Build env config
    cfg = TrainConfig(width=width, height=height, with_walls=True)
    env = SnakeRemoteEnv(
        address=cfg.address,
        width=cfg.width,
        height=cfg.height,
        with_walls=cfg.with_walls,
    )

    # Build network
    num_actions = 4
    policy_net = SnakeDQN(height, width, num_actions=num_actions).to(device)
    policy_net.load_state_dict(ckpt["model_state_dict"])
    policy_net.eval()

    def select_action(state: np.ndarray) -> int:
        # Greedy policy: no epsilon
        state_t = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_t)
        return int(q_values.argmax(dim=1).item())

    rewards = []
    foods_list = []

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        total_reward = 0.0
        foods_eaten = 0
        last_score = info.get("score", 0)

        done = False
        steps = 0

        while not done and steps < cfg.max_steps_per_episode:
            action = select_action(obs)
            next_obs, reward, done, info = env.step(action)

            total_reward += reward

            current_score = info.get("score", last_score)
            if current_score > last_score:
                foods_eaten += 1
            last_score = current_score

            obs = next_obs
            steps += 1

        rewards.append(total_reward)
        foods_list.append(foods_eaten)
        print(f"[Eval Ep {ep:3d}] reward={total_reward:.2f} foods={foods_eaten}")

    print("==== EVAL SUMMARY ====")
    print(f"Episodes: {num_episodes}")
    print(f"Avg reward: {np.mean(rewards):.2f}")
    print(f"Avg foods:  {np.mean(foods_list):.2f}")


if __name__ == "__main__":
    # Example: set via env, or hardcode here
    ckpt = os.getenv("EVAL_CHECKPOINT", "../models/checkpoints/snake_dqn_ep10000.pt")
    eval_checkpoint(ckpt, num_episodes=50)
