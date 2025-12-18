# python/trainer/PPO/model.py
# Placeholder for Actor-Critic network

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SnakeActorCritic(nn.Module):
    """
    Placeholder Actor-Critic network for PPO.
    Input: (batch, 4, H, W) one-hot grid channels for {empty, snake, food, wall}
    Outputs:
        - action_logits: logits over 4 actions
        - state_value: estimated value of the state
    """

    def __init__(self, height: int, width: int, num_actions: int = 4, in_channels: int = 4):
        super().__init__()
        self.height = height
        self.width = width
        self.num_actions = num_actions
        self.in_channels = in_channels

        # Simple CNN trunk
        self.conv == nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, height, width)
            n = self.conv(dummy).shape[-1]

        self.actor = nn.Sequential(
            nn.Linear(n, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(n, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the Actor-Critic network.
        
        Args:
            x: (batch, in_channels, height, width) tensor with one-hot encoded grid
            
        Returns:
            logits: (batch, num_actions) action logits
            value: (batch,) state value estimates
        """
        z = self.conv(x)
        logits = self.actor(z)
        value = self.critic(z).squeeze(-1)
        return logits, value