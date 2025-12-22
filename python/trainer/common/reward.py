#python/trainer/common/reward.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Union

DeathCause = Optional[Literal["unspecified", "wall", "self", "stall"]]

DeltaDist = Optional[Literal["closer", "farther", "same"]]

DeathCauseValue = Union[DeathCause, int]
@dataclass
class RewardConfig:

    # Death penalties
    death_wall: float = 8.0
    death_self: float = 8.0
    death_stall: float = 8.0

    # Hunger penalty applied after grace period
    hunger_grace: int = 10
    hunger_scale: float = 0.02
    hunger_max_penalty: float = 0.4

    # Step shaping
    step_penalty: float = 0.02
    step_no_change: float = 0.005
    step_closer_bonus: float = 0.03
    step_farther_penalty: float = 0.03

    # Food reward shaping
    food_base: float = 6.0
    food_scale: float = 0.5
    food_cap: Optional[float] = None

    def death_reward(self, cause: DeathCause) -> float:

        if isinstance(cause, int):
            cause = {0: "unspecified", 1: "wall", 2: "self", 3: "stall"}.get(cause, None)

        if cause == "wall":
            return -self.death_wall
        if cause == "self":
            return -self.death_self
        if cause == "stall":
            return -self.death_stall
        return 0.0
        
    def step_reward(self, dist: DeltaDist, steps_since_food: int) -> float:

        reward = -self.step_penalty

        if dist == "closer": # Moved closer
            reward += self.step_closer_bonus
        elif dist == "farther": # Moved farther away
            reward -= self.step_farther_penalty
        elif dist == "same": # Distance did not change
            reward -= self.step_no_change


        if (steps_since_food > self.hunger_grace):
            hunger = (steps_since_food - self.hunger_grace)
            extra_pen = min(self.hunger_max_penalty, hunger * self.hunger_scale)
            reward -= extra_pen

        return reward
        
    def food_reward(self, foods_eaten_in_episode: int) -> float:
        
        r = self.food_base + self.food_scale * float(foods_eaten_in_episode)
        if self.food_cap is not None:
            r = min(r, self.food_cap)
        return r
