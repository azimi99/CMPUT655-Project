import numpy as np
import glob

from matplotlib import pyplot as plt


def plot_all_rewards(base_dir, parameters, save_path):
    cur_dir = f"{base_dir}/"
    for i, (k, v) in enumerate(parameters.items()):
        cur_dir = cur_dir + f"{'_' if i != 0 else ''}{k}_{v}"
    
    cur_dir += "*"
    f_list = glob.glob(f"{cur_dir}/rewards.npy")
    ep_list = glob.glob(f"{cur_dir}/episode_ends.npy")
    results = []
    eps = []
    # mn = float("inf")
    for f, e in zip(f_list, ep_list):
        eps.append(np.load(e))
        results.append(np.load(f))
        # mn = min(results[-1].shape[0], mn)

    for i in range(len(results)):
        # results[i] = results[i][:mn]
        plt.plot(eps[i], results[i][eps[i]], alpha=0.3)
    
    # results = np.array(results)
    # mean_results = np.mean(results, axis=0)
    # plt.plot(mean_results, "blue")

    plt.savefig(save_path)


if __name__ == "__main__":
    res = plot_all_rewards("../../results/semi_grad_sarsa_without_eps_const/", {"lr": 0.001}, "../../imgs/semigrad_sarsa_without_eps_rewards_nochange.png")
    