# python/trainer/model.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SnakeDQN(nn.Module):
    """
    Simple convolutional network:
    Input: (batch, 1, H, W) grid {0,1,2,3}
    Output: Q-values over 4 actions
    """

    def __init__(self, height: int, width: int, num_actions: int = 4):
        super().__init__()
        self.height = height
        self.width = width
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        conv_out_dim = 32 * height * width

        self.fc1 = nn.Linear(conv_out_dim, 128)
        self.fc2 = nn.Linear(128, num_actions)

    def forward(self, x):
        """
        x: (batch, 1, H, W), float tensor
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
