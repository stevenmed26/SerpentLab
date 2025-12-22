# python/trainer/env_client.py

import grpc
import numpy as np

import trainer.env.snake_env_pb2 as pb
import trainer.env.snake_env_pb2_grpc as pb_grpc
from trainer.common.reward import RewardConfig

from typing import Optional

rcfg = RewardConfig()
reward = 0

class SnakeEnvClient:
    """
    Low-level client for the SnakeEnv gRPC service.
    The RL code can wrap this in a Gym-like API.
    """

    def __init__(self, address: str = "localhost:50051"):
        self._channel = grpc.insecure_channel(address)
        self._stub = pb_grpc.SnakeEnvStub(self._channel)
        self.session_id: str | None = None
        self.width: Optional[int] = None
        self.height: Optional[int] = None

    def reset(self, width: Optional[int] = None, height: Optional[int] = None, with_walls: bool = True):
        """
        Reset the environment. Optionally override width/height/with_walls.
        Returns: observation (np.ndarray), info dict
        """
        req = pb.ResetRequest(
            session_id=self.session_id or "",
            width=int(width or 0),
            height=int(height or 0),
            with_walls=bool(with_walls),
        )
        resp: pb.ResetResponse = self._stub.Reset(req)

        self.session_id = resp.session_id
        self.width = resp.width
        self.height = resp.height

        obs = self._grid_to_obs(resp.grid, resp.width, resp.height)
        info = {
            "score": resp.score,
            "done": resp.done,
        }
        return obs, info

    def step(self, action: int):
        """
        Take a step in the environment.

        action: int in {0, 1, 2, 3}
        Returns: obs (np.ndarray), reward (float), done (bool), info (dict)
        """
        if self.session_id is None:
            raise RuntimeError("Environment not reset; call reset() first.")

        req = pb.StepRequest(
            session_id=self.session_id,
            action=int(action),
        )
        resp: pb.StepResponse = self._stub.Step(req)

        obs = self._grid_to_obs(resp.grid, resp.width, resp.height)
        # reward = float(resp.reward)
        done = bool(resp.done)
        if resp.ate_food:
            reward = rcfg.food_reward(resp.score)
        elif done:
            reward = rcfg.death_reward(resp.death_cause)
        else:
            reward = rcfg.step_reward(resp.delta_dist, resp.steps_since_food)
        
        info = {
            "score": int(resp.score),
            "step_index": int(resp.step_index),
        }
        return obs, reward, done, info

    @staticmethod
    def _grid_to_obs(grid, width: int, height: int) -> np.ndarray:
        """
        Convert flattened int32 grid into a (H, W) int8 numpy array.
        Values:
          0 = empty, 1 = snake, 2 = food, 3 = wall
        """
        arr = np.asarray(grid, dtype=np.int8)
        assert arr.size == width * height, "grid size mismatch"
        return arr.reshape((height, width))

    def close(self):
        self._channel.close()
