import os
import time

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, optim

from envs import make_env
from utils import seed_everything


if __name__ == "__main__":
    #------------ Configs ------------#
    #TODO: Replace these with arguments at some point

    # Global configs
    seed = 20231124
    debug = True
    device = torch.device("cpu")

    # Environment configs
    env_name = "MiniGrid-DistShift1-v0"
    base_dir = f"./results/semi_grad_sarsa/{env_name}"
    normalize_reward = False
    gamma = 0.9
    
    # NN configs
    model_name = "sample"
    weight_path = f"{base_dir}/{model_name}.pth"
    lr = 1e-4
    
    # Experiment configurations
    timestep_limit = 1000 if debug else 500_000

    #------------ Training ------------#
    # Set seed for everything
    seed_everything(seed)
    
    # Create the necessary components
    env = make_env(env_name,
                   flat_obs=True,
                   normalize_reward=normalize_reward,
                   gamma=gamma,
                   )

    # Define components around model
    q_function = nn.Sequential(
        nn.Linear(env.base_env.observation_space.shape[0]//3, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 4)
    )

    q_function.load_state_dict(torch.load(weight_path))
    q_function.eval()

    optimizer = optim.Adam(q_function.parameters(), lr=lr)

    # Training algorithm
    timestep_count = 0
    ep_count = 1
    avg_rewards = []
    while True:
        # Initialize state
        obs, _ = env.reset(seed=seed)
        obs = obs.reshape((-1, 3))[:, 0].astype(np.float32)
        obs /= obs.max()
        obs = torch.Tensor(obs).to(device)
        
        # Take action based on the current q_function
        with torch.no_grad():
            action = q_function(obs).argmax().item()
            
        is_done = False
        st_time = time.time()
        prev_count = timestep_count
        r = 0
        while not is_done:
            # Step environment
            nxt_obs, reward, term, trunc, _ = env.step(action)
            nxt_obs = nxt_obs.reshape((-1, 3))[:, 0].astype(np.float32)
            nxt_obs /= nxt_obs.max()
            nxt_obs = torch.Tensor(nxt_obs).to(device)

            # Update flag for the termination
            is_done = term or trunc
            r += reward

            nxt_action = None
            with torch.no_grad():
                if not is_done:
                    # Will follow the greedy policy
                    q_pred_nxt = q_function(nxt_obs)
                    nxt_action = q_pred_nxt.argmax().item()

            obs = nxt_obs
            action = nxt_action

            timestep_count += 1
        
        # if ep_count % 10 == 0:
        #     print(f"Episode {ep_count} is done in {timestep_count - prev_count} steps, {time.time() - st_time} secs")
        ep_count += 1

        r /= (timestep_count - prev_count)
        avg_rewards.append(r)

        if timestep_count >= timestep_limit:
            break
    
    # Plot everything
    plt.plot(avg_rewards)
    plt.savefig(f"{base_dir}/avg_rewards_eval.png")