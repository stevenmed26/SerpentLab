# python/trainer/PPO/train_loop.py
# Placeholder for PPO training loop
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

from trainer.gym_env import SnakeRemoteEnv
from trainer.common.obs import one_hot_obs
from trainer.PPO.model import SnakeActorCritic
from trainer.PPO.rollout_buffer import RolloutBuffer


@dataclass
class PPOConfig:
    # Env
    address: str = os.getenv("TRAINER_ENV_ADDR", "localhost:50051")
    width: int = 10
    height: int = 10
    with_walls: bool = True

    # Training
    total_timesteps: int = 2_000_000
    n_steps: int = 256
    batch_size: int = 256
    n_epochs: int = 4

    gamma: float = 0.99
    gae_lambda: float = 0.95

    learning_rate: float = 2.5e-4
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef_start: float = 0.02
    entropy_coef_end: float = 0.005
    entropy_decay_updates: int = 2000
    max_grad_norm: float = 0.5

    # Logging / checkpoints
    log_interval_episodes: int = 50
    checkpoint_dir: str = "../python/models/PPO/checkpoints"
    checkpoint_interval_updates: int = 500
    resume_from: str = ""

def opposite_action(a: int) -> int:
    return (a + 2) % 4

def mask_logits_no_reverse(logits: torch.Tensor, last_action: Optional[int]) -> torch.Tensor:
    """
    logits: (1, num_actions)
    last_action: int or None
    """
    if last_action is None:
        return logits
    rev = opposite_action(last_action)
    masked = logits.clone()
    masked[..., rev] = -1e9
    return masked


def train_ppo(
    config: PPOConfig,
    stop_event=None,
    on_episode=None,
    set_status=None,
    device: Optional[str] = None,
    seed: int = 42,
):
    def entropy_coef(update_idx: int) -> float:
        if config.entropy_decay_updates <= 0:
            return config.entropy_coef_end
        frac = min(1.0, update_idx / config.entropy_decay_updates)
        return config.entropy_coef_start + frac * (config.entropy_coef_end - config.entropy_coef_start)
    
    # ----- device and seeds -----
    if device is None or device == "auto":
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cuda":
        if not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            device_t = torch.device("cpu")
        else:
            device_t = torch.device("cuda")
    elif device == "cpu":
        device_t = torch.device("cpu")
    else:
        raise ValueError(f"Unknown device: {device}")
    
    print(f"[device] requested={device} resolved={device_t} cuda_available={torch.cuda.is_available()}")



    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_t.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    print("device:", device_t, "cuda_available:", torch.cuda.is_available())
    if device_t.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ----- environment -----
    env = SnakeRemoteEnv(
        address=config.address,
        width=config.width,
        height=config.height,
        with_walls=config.with_walls,
    )

    #infer obs dims
    obs, info = env.reset()
    H, W = obs.shape
    C = 4
    num_actions = 4

    # ---- initialize movement ----
    last_action: Optional[int] = None
    steps_this_episode = 0

    # ----- model + optimizer-----
    model = SnakeActorCritic(height=H, width=W, num_actions=num_actions).to(device_t)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    #resume
    if config.resume_from:
        ckpt = torch.load(config.resume_from, map_location=device_t)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Resumed PPO from {config.resume_from}")

    # ----- rollout storage -----
    buf = RolloutBuffer(
        n_steps=config.n_steps,
        obs_shape=(C, H, W),
        device=device_t,
    )

    # episode tracking
    episode = 0
    episode_rewards = []
    ep_return = 0.0
    foods_eaten = 0
    last_score = info.get("score", 0)

    # training progress
    total_steps = 0
    update_idx = 0

    def finish_episode():
        nonlocal episode, ep_return, foods_eaten, last_score, steps_this_episode
        episode += 1
        episode_rewards.append(ep_return)
        avg_last_50 = (
            np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else float(np.mean(episode_rewards))
        )
        
        if set_status:
            set_status(
                {
                    "episode": episode,
                    "last_reward": float(ep_return),
                    "avg_last_50": avg_last_50,
                    "foods": foods_eaten,
                    "steps_episode": int(steps_this_episode),
                }
            )

        if on_episode:
            on_episode(
                {
                    "episode": episode,
                    "reward": float(ep_return),
                    "avg_last_50": avg_last_50,
                    "steps_taken": int(total_steps),
                    "foods": int(foods_eaten),
                    "update": int(update_idx),
                }
            )

        if config.log_interval_episodes and (episode % config.log_interval_episodes == 0):
            print(
                f"[Episode {episode:4d}] "
                f"[reward={ep_return:.2f}] "
                f"[avg_last_50={avg_last_50:.2f}] "
                f"[foods={foods_eaten}] "
                f"[steps_total={total_steps}] "
                f"[update={update_idx}] "
                f"[steps_ep={steps_this_episode}]"
            )

        ep_return = 0.0
        foods_eaten = 0
        last_score = 0
        steps_this_episode = 0

    # reset episode state cleanly
    obs, info = env.reset()
    last_score = info.get("score", 0)

    while total_steps < config.total_timesteps:
        if stop_event is not None and stop_event.is_set():
            print("Training stopped by stop event.")
            break

        obs_oh = one_hot_obs(obs, num_channels=C)  # (C,H,W)
        obs_t = torch.from_numpy(obs_oh).unsqueeze(0).to(device_t)  # (1,C,H,W)

        with torch.no_grad():
            logits, value_t = model(obs_t)
            logits = mask_logits_no_reverse(logits, last_action)
            dist = Categorical(logits=logits)
            action_t = dist.sample()
            logprob_t = dist.log_prob(action_t)

        action = int(action_t.item())
        last_action = action
        value = float(value_t.squeeze(0).item())
        logprob = float(logprob_t.squeeze(0).item())

        next_obs, reward, done, info = env.step(action)

        # episode bookkeeping
        ep_return += float(reward)
        steps_this_episode += 1

        current_score = info.get("score", last_score)
        if current_score > last_score:
            foods_eaten += 1
        last_score = current_score

        # store transition
        buf.add(
            obs=obs_oh,
            action=action,
            reward=float(reward),
            done=bool(done),
            value=value,
            logprob=logprob,
        )

        total_steps += 1
        obs = next_obs
        last_action = action

        if done:
            finish_episode()
            obs, info = env.reset()
            last_score = info.get("score", 0)
            last_action = None
            steps_this_episode = 0

        if not buf.full:
            continue

        if total_steps >= config.total_timesteps:
            break

        # bootstrap value for GAE
        model.eval()
        obs_oh2 = one_hot_obs(obs, num_channels=C)  # (C,H,W)
        obs_t2 = torch.from_numpy(obs_oh2).unsqueeze(0).to(device_t)  # (1,C,H,W)
        with torch.no_grad():
            _logits2, last_value_t = model(obs_t2)
        
        last_done = bool(buf.dones[buf.n_steps - 1])
        last_value = 0.0 if last_done else float(last_value_t.squeeze(0).item())

        advantages, returns = buf.compute_gae(
            last_value=last_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        # normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        batch = buf.to_torch(advantages=advantages, returns=returns)

        # ----- PPO update -----
        model.train()
        update_idx += 1

        B = config.n_steps
        idxs = np.arange(B)

        for _epoch in range(config.n_epochs):
            np.random.shuffle(idxs)

            for start in range(0, B, config.batch_size):
                mb_idxs = idxs[start:start + config.batch_size]

                mb_obs = batch.obs[mb_idxs]
                mb_actions = batch.actions[mb_idxs]
                mb_old_logprobs = batch.logprobs[mb_idxs]
                mb_returns = batch.returns[mb_idxs]
                mb_adv = batch.advantages[mb_idxs]
                mb_old_values = batch.values[mb_idxs]


                logits, values = model(mb_obs)
                dist = Categorical(logits=logits)

                new_logprobs = dist.log_prob(mb_actions)
                entropy_t = dist.entropy()

                # policy ratio
                ratio = torch.exp(new_logprobs - mb_old_logprobs)

                # clipped objective
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                # value loss
                # simple MSE between returns and value estimates
                value_loss = F.mse_loss(values, mb_returns)

                ent_coef = entropy_coef(update_idx)
                loss = policy_loss + config.value_coef * value_loss - ent_coef * entropy_t.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

        buf.reset()
                
        # ----- checkpoints -----
        if (config.checkpoint_interval_updates > 0 and
            update_idx % config.checkpoint_interval_updates == 0):
            ckpt_path = os.path.join(
                config.checkpoint_dir,
                f"ppo_snake_update{update_idx}.pt",
            )
            config_dict = dict(config.__dict__)
            config_dict["height"] = H
            config_dict["width"] = W

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "steps_total": total_steps,
                    "episode": episode,
                    "update_idx": update_idx,
                    "config": config_dict,
                    "avg_reward_last_100": np.mean(episode_rewards[-100:]) if episode_rewards else 0.0,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")

    env.close()