from envs.envs import make_env
from utils import seed_everything
from nns.nns import EMLP, MLP

import torch
import numpy as np
import matplotlib.pyplot as plt


def format_observation(obs):
    obs = obs.reshape((-1, 3))[:, 0].astype(np.float32)
    obs /= obs.max()
    return obs


def epsilon_greedy(q, epsilon):
    prob = np.random.uniform()
    if prob < epsilon:
        return np.random.randint(0, len(q))
    else:
        return torch.argmax(q)


class Agent():
    def __init__(self, dims, alpha, gamma, epsilon=0.1):
        self.q_net = EMLP(dims=dims)
        self.gamma = gamma
        self.epsilon = epsilon
        self.optimzer = torch.optim.Adam(self.q_net.parameters(), lr=alpha)
    
    def get_action(self, state, e_greedy=False):
        qs = self.q_net(state)
        if e_greedy == True:
            return epsilon_greedy(qs, self.epsilon)
        else:
            return torch.argmax(qs).item()

    def update(self, state, action, reward, next_state, next_action, terminal):
        target = reward + self.gamma * self.q_net(next_state)[next_action].detach() * (1-terminal)
        q = self.q_net(state)[action]
        loss = 1/2 * torch.square(target - q)

        self.optimzer.zero_grad()
        loss.backward()
        self.optimzer.step()
            

def experiment_run(env_names):
    #------------ Configs ------------#
    for env_name in env_names:
        # Global configs
        seed = 20231124
        
        # Environment configs
        normalize_reward = False
        PENALIZE_DEATH = True
        
        # NN configs
        use_nn = False
        
        # Algorithm configs
        algo_name = "sarsa"
        num_episodes = 1000
        LEARNING_RATE = 0.01
        GAMMA = 0.99

        #------------ Training ------------#
        # Set seed for everything
        seed_everything(seed)
        
        # Create the necessary components
        env = make_env(env_name,flat_obs=True, \
            penalize_death=PENALIZE_DEATH, normalize_reward=normalize_reward, gamma=GAMMA)
        dims = [env.base_env.observation_space.shape[0]//3, 100, 100, 4]
        agent = Agent(dims=dims, alpha=LEARNING_RATE, gamma=GAMMA)
        avg_returns = []
        reward_accumulator = 0
        episode_length = 1
        terminal = False
        # Train
        for i in range(num_episodes):
            obs, _ = env.reset(seed=seed)
            obs = torch.Tensor(format_observation(obs))
            action = agent.get_action(obs, e_greedy=True)
            while not terminal and episode_length < 5000: # truncate long episodes
                next_obs, reward, terminal, _, _ = env.step(action)
                next_obs = torch.Tensor(format_observation(next_obs))
                if terminal:
                   reward_accumulator = reward

                episode_length += 1
                next_action = agent.get_action(next_obs, e_greedy=True)
                agent.update(obs, action, reward, next_obs, next_action, terminal)

                obs = next_obs
                action = next_action
            print(f"episode {i+1} has completed. #Steps: {episode_length}")
            avg_returns.append(reward_accumulator)
            episode_length = 1
            reward_accumulator = 0
            terminal = False
        print(f"Completed training on {env_name}")
        plt.figure()
        plt.plot(avg_returns)
        plt.show()
        plt.savefig(f"../results/returns-{env_name}-\
            {seed}-{'normalized' if normalize_reward else 'raw'}")    
    


if __name__ == "__main__":
    # environments = [
    #                 "MiniGrid-Empty-Random-5x5-v0",
    #                 "MiniGrid-DistShift1-v0", 
    #                 "MiniGrid-LavaGapS5-v0"
    #                 ]
    environments = ["MiniGrid-Empty-5x5-v0"]
    experiment_run(environments)