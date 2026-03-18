import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import matplotlib.pyplot as plt

from config import Config, setup
from agents import DQNAgent
from train import train_dqn


def main():
    config = Config()
    device = setup(config)

    env = gym.make(config.env_id)
    env.reset(seed=config.seed)
    env.action_space.seed(config.seed)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size, device, config, network_type="standard")
    _, epoch_scores = train_dqn(env, agent, config, num_episodes=500)

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(epoch_scores)), epoch_scores)
    plt.title('DQN on Acrobot-v1: Baseline Performance')
    plt.xlabel('Epoch (10 episodes)')
    plt.ylabel('Average Score')
    plt.axhline(y=-100, color='r', linestyle='--', label='Solving threshold (-100)')
    plt.legend()
    plt.savefig('dqn_acrobot_baseline.png')
    plt.show()

    env.close()


if __name__ == '__main__':
    main()
