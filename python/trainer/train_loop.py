# python/trainer/train_loop.py

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from gym_env import SnakeRemoteEnv
from model import SnakeDQN
from replay_buffer import ReplayBuffer, Transition


@dataclass
class TrainConfig:
    address: str = os.getenv("TRAINER_ENV_ADDR", "localhost:50051")
    width: int = 10
    height: int = 10
    with_walls: bool = True

    num_episodes: int = 10_000
    max_steps_per_episode: int = 500

    batch_size: int = 64
    gamma: float = 0.99

    # Epsilon-greedy
    eps_start: float = 1.0
    eps_end: float = 0.10
    eps_decay_episodes: int = 2500

    buffer_capacity: int = 100_000
    learning_rate: float = 1e-3
    target_update_interval: int = 50  # episodes

    checkpoint_dir: str = "../models/checkpoints"
    checkpoint_interval: int = 1000  # episodes

    resume_from: str = ""


def linear_epsilon(config: TrainConfig, episode: int) -> float:
    if episode >= config.eps_decay_episodes:
        return config.eps_end
    fraction = episode / config.eps_decay_episodes
    return config.eps_start + fraction * (config.eps_end - config.eps_start)


def train(config: TrainConfig, stop_event=None, on_episode=None, set_status=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = SnakeRemoteEnv(
        address=config.address,
        width=config.width,
        height=config.height,
        with_walls=config.with_walls,
    )

    # Get initial observation to infer dimensions.
    obs, _ = env.reset()
    height, width = obs.shape
    num_actions = 4

    policy_net = SnakeDQN(height, width, num_actions=num_actions).to(device)
    target_net = SnakeDQN(height, width, num_actions=num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=config.learning_rate)
    buffer = ReplayBuffer(capacity=config.buffer_capacity)

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    def select_action(state: np.ndarray, eps: float) -> int:
        if np.random.rand() < eps:
            return np.random.randint(num_actions)

        state_t = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def optimize_model():
        if len(buffer) < config.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = buffer.sample(config.batch_size)

        states_t = torch.from_numpy(states).float().unsqueeze(1).to(device)       # (B,1,H,W)
        next_states_t = torch.from_numpy(next_states).float().unsqueeze(1).to(device)
        actions_t = torch.from_numpy(actions).long().unsqueeze(1).to(device)      # (B,1)
        rewards_t = torch.from_numpy(rewards).float().to(device)                  # (B,)
        dones_t = torch.from_numpy(dones).float().to(device)                      # (B,)

        # Q(s,a)
        q_values = policy_net(states_t).gather(1, actions_t).squeeze(1)

        # Q_target(s', a') using target_net (max over actions)
        with torch.no_grad():

            # Online picks next actions
            next_q_online = policy_net(next_states_t)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)

            next_q_values = target_net(next_states_t).gather(1, next_actions).squeeze(1)
            targets = rewards_t + config.gamma * next_q_values * (1.0 - dones_t)

        loss = F.mse_loss(q_values, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        optimizer.step()

        return float(loss.item())

    episode_rewards = []

    for episode in range(1, config.num_episodes + 1):
        if stop_event is not None and stop_event.is_set():
            break

        obs, info = env.reset()
        total_reward = 0.0
        foods_eaten = 0
        last_score = info.get("score", 0)

        eps = linear_epsilon(config, episode)

        for step in range(config.max_steps_per_episode):
            if stop_event is not None and stop_event.is_set():
                break
            action = select_action(obs, eps)
            next_obs, reward, done, info = env.step(action)

            total_reward += reward
            current_score = info.get("score", last_score)
            if current_score > last_score:
                foods_eaten += 1
            last_score = current_score

            buffer.push(
                Transition(
                    state=obs,
                    action=action,
                    reward=reward,
                    next_state=next_obs,
                    done=done,
                )
            )

            obs = next_obs

            loss_val = optimize_model()

            if done:
                break

        episode_rewards.append(total_reward)
        avg_last_50 = float(np.mean(episode_rewards[-50:])) if len(episode_rewards) >= 50 else float(np.mean(episode_rewards))

        if set_status:
            set_status(episode=episode, last_reward=float(total_reward), avg_last_50=avg_last_50, foods=foods_eaten)

        if on_episode:
            on_episode({
                "episode": episode,
                "eps": float(eps),
                "reward": float(total_reward),
                "avg_last_50": avg_last_50,
                "foods": foods_eaten,
                "buffer": int(len(buffer)),
            })

        # Update target net
        if episode % config.target_update_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Checkpoint
        if episode % config.checkpoint_interval == 0:
            ckpt_path = os.path.join(config.checkpoint_dir, f"snake_dqn_ep{episode}.pt")
            config_dict = dict(config.__dict__)
            config_dict["width"] = width
            config_dict["height"] = height

            torch.save(
                {
                    "episode": episode,
                    "model_state_dict": policy_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config_dict,
                    "avg_reward_last_100": float(np.mean(episode_rewards[-100:])),
                    "foods_eaten": foods_eaten,
                },
                ckpt_path,
            )

        if episode % 50 == 0:
            avg_last_50 = float(np.mean(episode_rewards[-50:])) if len(episode_rewards) >= 50 else float(
                np.mean(episode_rewards)
            )
            print(
                f"[Episode {episode:4d}] "
                f"eps={eps:.3f} "
                f"reward={total_reward:.2f} "
                f"foods={foods_eaten} "
                f"avg_last_50={avg_last_50:.2f} "
                f"buffer={len(buffer)}"
            )

    env.close()
