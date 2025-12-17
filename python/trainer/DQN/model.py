# python/trainer/model.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SnakeDQN(nn.Module):
    """
    Simple convolutional network:
    Input: (batch, 4, H, W) one-hot grid channels for {empty, snake, food, wall}
    Output: Q-values over 4 actions
    """

    def __init__(self, height: int, width: int, num_actions: int = 4):
        super().__init__()
        self.height = height
        self.width = width
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        conv_out_dim = 64 * height * width

        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, 512),
            nn.ReLU(),
        )

        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, num_actions)

    def forward(self, x):
        """
        x: (batch, 4, H, W), float tensor with one-hot channels
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        v = self.value(x)
        a = self.advantage(x)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
