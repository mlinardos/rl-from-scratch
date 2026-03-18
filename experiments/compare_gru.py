import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import matplotlib.pyplot as plt

from config import Config, setup
from agents import DQNAgent, GRUDQNAgent
from train import train_dqn, train_gru_dqn


def main():
    config = Config()
    device = setup(config)

    env = gym.make(config.env_id)
    env.reset(seed=config.seed)
    env.action_space.seed(config.seed)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    print("--- Running standard DQN ---")
    dqn_agent = DQNAgent(state_size, action_size, device, config, network_type="standard")
    _, dqn_epochs = train_dqn(env, dqn_agent, config, num_episodes=500)

    print("\n--- Running GRU DQN ---")
    gru_agent = GRUDQNAgent(state_size, action_size, device, config)
    _, gru_epochs = train_gru_dqn(env, gru_agent, config, num_episodes=500)

    plt.figure(figsize=(12, 8))
    plt.plot(range(len(dqn_epochs)), dqn_epochs, label='Standard DQN')
    plt.plot(range(len(gru_epochs)), gru_epochs, label='GRU DQN')
    plt.title('DQN vs GRU-DQN on Acrobot-v1')
    plt.xlabel('Epoch (10 episodes)')
    plt.ylabel('Average Score')
    plt.axhline(y=-100, color='r', linestyle='--', label='Solving threshold (-100)')
    plt.legend()
    plt.savefig('dqn_vs_gru_dqn_acrobot.png')
    plt.show()

    env.close()


if __name__ == '__main__':
    main()
