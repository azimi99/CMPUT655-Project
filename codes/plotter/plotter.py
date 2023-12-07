import copy
import sys

import numpy as np
import glob
from matplotlib import pyplot as plt

# from metrics import incremental_coverage


def incremental_coverage(states, total_states: int):
    if len(states) == 0:
        return 0
    
    visited = [False] * total_states
    coverage = np.zeros(len(states))
    coverage[0] = 1
    visited[states[0]] = True
    cnt = 1
    for i, state in enumerate(states[1:]):
        if not visited[state]:
            cnt += 1
            visited[state] = True
        coverage[i+1] = cnt

    return coverage / total_states


def formulate_path(base_dir, parameters):
    cur_dir = f"{base_dir}/"
    for i, (k, v) in enumerate(parameters.items()):
        cur_dir = cur_dir + f"{'_' if i != 0 else ''}{k}_{v}"
    cur_dir += "*"

    return cur_dir


def plot_all_rewards(base_dir, parameters, save_path):
    # Get output directory
    cur_dir = formulate_path(base_dir, parameters)

    f_list = glob.glob(f"{cur_dir}/rewards.npy")
    ep_list = glob.glob(f"{cur_dir}/episode_ends.npy")
    results = []
    eps = []
    for f, e in zip(f_list, ep_list):
        eps.append(np.load(e))
        results.append(np.load(f))
        # mn = min(results[-1].shape[0], mn)

    for i in range(len(results)):
        # results[i] = results[i][:mn]
        plt.plot(eps[i], results[i][eps[i]], alpha=0.3)

    plt.savefig(save_path)


def plot_incremental_coverage(base_dir, parameters, total_states, save_path):
    cur_dir = formulate_path(base_dir, parameters)
    
    f_list = glob.glob(f"{cur_dir}/states.npy")
    coverage = []
    for f in f_list:
        coverage.append(incremental_coverage(np.load(f), total_states))
    
    for c in coverage:
        plt.plot(c, alpha=0.4)
    
    plt.savefig(save_path)


if __name__ == "__main__":
    plot_incremental_coverage("../../results/semi_grad_sarsa_without_eps_6x6/", {"lr": 0.001}, 36, "../../imgs/semigrad_sarsa_without_eps_coverage_6x6.png")
    