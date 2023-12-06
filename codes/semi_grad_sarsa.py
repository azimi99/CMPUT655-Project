import argparse
import copy
import os
import time

import numpy as np
import torch
from torch import nn, optim

from envs import make_env
from metrics import Measurements
from utils import seed_everything, read_config, hash_env


def parse_arguments():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("-p", "--path", type=str, help="Path to a config file.")

    args = arg_parser.parse_args()
    
    return args


def run_single_config(config, save_dir):
    # Set a seed
    seed_everything(config["seed"])

    # Create logging class
    measurements = Measurements()

    # Create environment instance
    env = make_env(config["env_name"],
                flat_obs=config["flat_obs"],
                normalize_reward=config["normalize_reward"],
                penalize_death=config["penalize_death"],
                reward_amplification=config["reward_amplification"],
                mx_reward=config["mx_reward"],
                gamma=config["gamma"],
                )
    
    # Create model instance
    q_function = nn.Sequential(
        nn.Linear(env.base_env.observation_space.shape[0]//3, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 4)
    )
    optimizer = optim.Adam(q_function.parameters(), lr=config["lr"])

    # Training algorithm
    timestep_count = 0
    ep_count = 1
    width = env.base_env.unwrapped.width
    while True:
        # Initialize state
        obs, _ = env.reset(seed=config["seed"])
        cur_pos = env.base_env.unwrapped.agent_pos
        obs = obs.reshape((-1, 3))[:, 0].astype(np.float32)
        obs /= obs.max()
        obs = torch.Tensor(obs).to(config["device"])
        
        # Take action based on the current q_function
        with torch.no_grad():
            # actions.append(q_function(obs))
            if np.random.rand() < config["eps"]:
                action = np.random.randint(0, 4)
            else:
                action = q_function(obs).argmax().item()
            
        is_done = False
        st_time = time.time()
        prev_count = timestep_count
        while not is_done:
            # Step environment
            nxt_obs, reward, term, trunc, _ = env.step(action)
            nxt_obs = nxt_obs.reshape((-1, 3))[:, 0].astype(np.float32)
            nxt_obs /= nxt_obs.max()
            nxt_obs = torch.Tensor(nxt_obs).to(config["device"])

            # Update flag for the termination
            is_done = term or trunc

            nxt_action = None
            with torch.no_grad():
                # Compute base delta
                q_val = q_function(obs)[action]
                delta = reward - q_val
                
                if not is_done:
                    # Will follow the greedy policy
                    if np.random.rand() < config["eps"]:
                        nxt_action = np.random.randint(0, 4)
                    else:
                        q_pred_nxt = q_function(nxt_obs)
                        nxt_action = q_pred_nxt.argmax().item()
                    delta += config["gamma"] * q_pred_nxt[nxt_action]
                
            # Update weights
            loss = -delta * q_function(obs)[action]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            measurements.add_measurements(hash_env(cur_pos, width),
                                          action,
                                          reward,
                                          timestep_count if is_done else None)

            obs = nxt_obs
            cur_pos = env.base_env.unwrapped.agent_pos
            action = nxt_action

            timestep_count += 1
        
        if config["verbose"] > 0 and ep_count % config["verbose"] == 0:
            print(f"Episode {ep_count} is done in {timestep_count - prev_count} steps, {time.time() - st_time:.2f} secs")
        ep_count += 1

        if timestep_count >= config["total_steps"]:
            break
    
    # Save everything
    measurements.save_everything(save_dir)


if __name__ == "__main__":
    # Read configs
    args = parse_arguments()
    config = read_config(args.path)
    config["device"] = torch.device(config["device"])

    #------------ Training ------------#
    # Produce seeds for sweeping
    np.random.seed(config["base_seed"])
    seeds_for_sweep = [np.random.randint(10_000_000) for _ in range(config["n_seeds"])]

    # Create directory for saving the results
    base_dir = config["base_save_dir"]
    _, cur_dir = os.path.split(os.getcwd())
    if cur_dir == "codes" and base_dir[:2] != "..":
        base_dir = "../" + base_dir
        
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)

    # Prepare the combinations of sweeping variables
    # NOTE: CUSTOMIZE THE OUTER LOOP AND FOLDER CREATION PROCESS DEPENDING ON THE SWEEPING VARIABLES
    lrs = copy.deepcopy(config["lr"])
    run_cnt = 1
    run_start_time = time.time()
    for lr in lrs:
        for seed in seeds_for_sweep:
            # Overwrite the config
            config["lr"] = lr
            config["seed"] = seed

            save_dir = base_dir + f"/lr_{lr}_seed_{seed}"       # CHANGE THIS ACCORDING TO YOUR SWEEPING VARIABLES
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            # Start timer
            run_single_config(config, save_dir)
            runtime = time.time() - run_start_time
            
    mins = int(runtime) // 60
    secs = runtime - (mins * 60)
    print(f"Run {run_cnt} : {mins} mins {secs:.2f} secs")