from typing import Any

from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np


class CustomMinigridEnv():
    def __init__(self, base_env: Env):
        self.base_env = base_env
        
        # Action indices
        self.right = 0
        self.down = 1
        self.left = 2
        self.up = 3

        # Command in base environment
        self.face_left = 0
        self.face_right = 1
        self.forward = 2

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ):
        return self.base_env.reset(seed=seed, options=options)
        

    def step(self, action):
        self.adjust_direction(action)
        return self.base_env.step(self.forward)

    def adjust_direction(self, action):
        action_type, n_action = None, 0
        cur_dir = self.base_env.unwrapped.agent_dir

        if abs(cur_dir - action) == 2:
            action_type = self.face_right
            n_action = 2
        elif (cur_dir + 1) % 4 == action:
            action_type = self.face_right
            n_action = 1
        elif (cur_dir - 1) % 4 == action:
            action_type = self.face_left
            n_action = 1

        for i in range(n_action):
            self.uncount_action(action_type)

    def uncount_action(self, action):
        self.base_env.unwrapped.step_count -= 1
        self.base_env.step(action)


class FlattenObservation(gym.ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)

        imgSpace = env.observation_space.spaces["image"]
        imgSize = np.prod(imgSpace.shape)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize,),
            dtype="uint8",
        )

    def observation(self, observation):
        image = observation["image"]
        return image.flatten()


class TraditionalFlattenObservation(gym.ObservationWrapper):
    # Older version of observation flattening: 
    # Rendered from here: https://goodboychan.github.io/python/pytorch/reinforcement_learning/2020/08/06/03-Policy-Gradient-With-Gym-MiniGrid.html
    def __init__(self, env: Env):
        super().__init__(env)

        imgSpace = env.observation_space.spaces["image"]
        imgSize = np.prod(imgSpace.shape)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize,),
            dtype="uint8",
        )

    def observation(self, observation):
        unwrapped = self.env.unwrapped
        full_grid = unwrapped.grid.encode()
        full_grid[unwrapped.agent_pos[0]][unwrapped.agent_pos[1]] = np.array([
            OBJECT_TO_IDX["agent"],
            COLOR_TO_IDX["red"],
        ])
        full_grid = full_grid[1:-1, 1:-1]
        
        full_grid = full_grid.ravel()
        return full_grid


class NormalizeReward(gym.RewardWrapper):
    def __init__(self, env: Env, mx_reward: float = 1., gamma: float = 1.):
        super().__init__(env)
        
        self.mx_reward = abs(mx_reward)
        self.gamma = gamma

    def reward(self, reward: float) -> float:
        return reward / self.mx_reward + (self.gamma - 1.)
