# python/trainer/gym_env.py

from typing import Tuple, Dict, Any

import numpy as np

from .env_client import SnakeEnvClient


class SnakeRemoteEnv:
    """
    Minimal Gym-like wrapper:
      obs, info = env.reset()
      obs, reward, done, info = env.step(action)
    """

    def __init__(self, address: str = "localhost:50051", width: int = 10, height: int = 10, with_walls: bool = True):
        self.client = SnakeEnvClient(address=address)
        self.width = width
        self.height = height
        self.with_walls = with_walls

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.client.reset(width=self.width, height=self.height, with_walls=self.with_walls)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        return self.client.step(action)

    def close(self):
        self.client.close()
