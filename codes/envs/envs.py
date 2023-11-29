import gymnasium as gym
from gymnasium.core import Env
import minigrid
from minigrid.wrappers import FullyObsWrapper

from .custom_minigrid_wrappers import (
    NormalizeReward,
    FlattenObservation,
    TraditionalFlattenObservation,
    CustomMinigridEnv,
)

def make_env(
    env_name: str,
    flat_obs: bool = True,
    normalize_reward: bool = False,
    mx_reward: float = 1.0,
    gamma: float = 1.0,
):
    base_env = gym.make(env_name)
    if flat_obs:
        base_env = TraditionalFlattenObservation(base_env)
    if normalize_reward:
        base_env = NormalizeReward(base_env, mx_reward=mx_reward, gamma=gamma)

    # Wrap with custom action sequences
    env = CustomMinigridEnv(base_env)
    return env
