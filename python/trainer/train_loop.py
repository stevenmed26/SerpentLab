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
from collections import deque


@dataclass
class TrainConfig:
    address: str = os.getenv("TRAINER_ENV_ADDR", "localhost:50051")
    width: int = 10
    height: int = 10
    with_walls: bool = True

    num_episodes: int = 10_000
    max_steps_per_episode: int = 500

    batch_size: int = 128
    gamma: float = 0.99
    n_step: int = 3

    #PER
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_anneal_steps: int = 1_000_000

    # Epsilon-greedy
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 5000

    buffer_capacity: int = 100_000
    learning_rate: float = 5e-4
    target_update_interval: int = 50  # episodes

    checkpoint_dir: str = "../models/checkpoints"
    checkpoint_interval: int = 1000  # episodes

    resume_from: str = ""


def linear_epsilon(config: TrainConfig, episode: int) -> float:
    if episode >= config.eps_decay_episodes:
        return config.eps_end
    fraction = episode / config.eps_decay_episodes
    return config.eps_start + fraction * (config.eps_end - config.eps_start)

def one_hot_obs(obs: np.ndarray, num_channels: int = 4) -> np.ndarray:
    """
    obs: (H, W) with values in {0,1,2,3}
    returns: (C, H, W)
    """
    h, w = obs.shape
    out = np.zeros((num_channels, h, w), dtype=np.float32)
    for c in range(num_channels):
        out[c] = (obs == c)
    return out

    



def train(config: TrainConfig, stop_event=None, on_episode=None, set_status=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device, "cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

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
    buffer = ReplayBuffer(
        capacity=config.buffer_capacity,
        alpha=config.per_alpha,
    )

    nstep_buf = deque(maxlen=config.n_step)
    global_step = 0

    def current_beta(step: int) -> float:
        if config.per_beta_anneal_steps <= 0:
            return config.per_beta_end
        frac = min(1.0, step / config.per_beta_anneal_steps)
        return config.per_beta_start + frac * (config.per_beta_end - config.per_beta_start)

    def push_n_step(state, action, reward, next_state, done):
        nstep_buf.append((state, action, float(reward), next_state, bool(done)))

        if len(nstep_buf) < config.n_step and not done:
            return
        
        R = 0.0
        done_n = False
        next_state_n = nstep_buf[-1][3]
        steps_used = 0

        for i, (_, _, r, ns, d) in enumerate(nstep_buf):
            R += (config.gamma ** i) * float(r)
            steps_used = i + 1
            if d:
                done_n = True
                next_state_n = ns
                break

        s0, a0, _, _, _ = nstep_buf[0]
        gamma_n = config.gamma ** steps_used

        buffer.push(Transition(
            state=s0,
            action=a0,
            reward=R,
            next_state=next_state_n,
            done=done_n,
            gamma_n=gamma_n,
        ))

        if done_n:
            # flush remaining partial transitions
            while len(nstep_buf) > 1:
                nstep_buf.popleft()

                R = 0.0
                done_flush = False
                next_state_n = nstep_buf[-1][3]
                steps_used = 0

                for i, (_, _, r, ns, d) in enumerate(nstep_buf):
                    R += (config.gamma ** i) * float(r)
                    steps_used = i + 1
                    if d:
                        done_flush = True
                        next_state_n = ns
                        break

                s0, a0, _, _, _ = nstep_buf[0]
                gamma_n = config.gamma ** steps_used

                buffer.push(Transition(
                    state=s0,
                    action=a0,
                    reward=R,
                    next_state=next_state_n,
                    done=done_flush,
                    gamma_n=gamma_n,
                ))

            nstep_buf.clear()
            return
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    def select_action(state: np.ndarray, eps: float) -> int:
        if np.random.rand() < eps:
            return np.random.randint(num_actions)

        state_oh = one_hot_obs(state)  # (C,H,W)
        state_t = torch.from_numpy(state_oh).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def optimize_model():
        if len(buffer) < config.batch_size:
            return 0.0

        beta = current_beta(global_step)
        states, actions, rewards, next_states, dones, gammas, idxs, weights = buffer.sample(config.batch_size, beta=beta)

        states_oh = np.stack([one_hot_obs(s) for s in states], axis=0)         # (B,C,H,W)
        next_states_oh = np.stack([one_hot_obs(s) for s in next_states], axis=0)

        states_t = torch.from_numpy(states_oh).float().to(device)       # (B,C,H,W)
        next_states_t = torch.from_numpy(next_states_oh).float().to(device)

        actions_t = torch.from_numpy(actions).long().unsqueeze(1).to(device)      # (B,1)
        rewards_t = torch.from_numpy(rewards).float().to(device)                  # (B,)
        dones_t = torch.from_numpy(dones).float().to(device)                      # (B,)
        gammas_t = torch.from_numpy(gammas).float().to(device)                  # (B,)
        weights_t = torch.from_numpy(weights).float().to(device)                # (B,)

        scaler = torch.cuda.amp.GradScaler()

        # Q(s,a)
        q_values = policy_net(states_t).gather(1, actions_t).squeeze(1)

        # Q_target(s', a') using target_net (max over actions)
        with torch.no_grad():

            # Online picks next actions
            next_q_online = policy_net(next_states_t)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)

            next_q_values = target_net(next_states_t).gather(1, next_actions).squeeze(1)

            targets = rewards_t + gammas_t * next_q_values * (1.0 - dones_t)

        td_errors = (q_values - targets).detach().abs()
        loss_per_item = F.smooth_l1_loss(q_values, targets, reduction='none')
        loss = (weights_t * loss_per_item).mean()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        tau = 0.005
        with torch.no_grad():
            for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
                tp.data.mul_(1 - tau).add_(tau * pp.data)

        new_prios = (td_errors.cpu().numpy() + 1e-6)
        buffer.update_priorities(idxs, new_prios)

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

            global_step += 1

            total_reward += reward
            current_score = info.get("score", last_score)
            if current_score > last_score:
                foods_eaten += 1
            last_score = current_score

            push_n_step(obs, action, reward, next_obs, done)

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
