import numpy as np


class Measurements:
    def __init__(self):
        self.episode_ends = [] 
        self.states = []
        self.actions = []
        self.rewards = []

    def save_everything(self, save_dir):
        np.save(f"{save_dir}/episode_ends.npy", self.episode_ends)
        np.save(f"{save_dir}/states.npy", self.states)
        np.save(f"{save_dir}/actions.npy", self.actions)
        np.save(f"{save_dir}/rewards.npy", self.rewards)

    def add_measurements(
        self,
        state,
        action,
        reward,
        episode_end = None,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        if episode_end:
            self.episode_ends.append(episode_end)


def incremental_coverage(states, total_states: int):
    if len(states) == 0:
        return 0
    
    visited = [False] * total_states
    incremental_coverage = np.zeros(len(states))
    incremental_coverage[0] = 1
    visited[states[0]] = True
    cnt = 1
    for i, state in enumerate(states[1:]):
        if not visited[state]:
            cnt += 1
            visited[state] = True
        incremental_coverage[i] = cnt
    
    return incremental_coverage / total_states


def coverage(states, total_states: int):
    return len(np.unique(states)) / total_states