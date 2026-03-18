import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
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

    print("--- Running standard DQN ---")
    dqn_agent = DQNAgent(state_size, action_size, device, config, network_type="standard")
    _, dqn_epochs = train_dqn(env, dqn_agent, config, num_episodes=500)

    print("\n--- Running Dueling DQN ---")
    dueling_agent = DQNAgent(state_size, action_size, device, config, network_type="dueling")
    _, dueling_epochs = train_dqn(env, dueling_agent, config, num_episodes=500)

    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(12, 8))
    plt.plot(range(len(dqn_epochs)), dqn_epochs, label='Standard DQN')
    plt.plot(range(len(dueling_epochs)), dueling_epochs, label='Dueling DQN')
    plt.title('DQN vs Dueling DQN on Acrobot-v1')
    plt.xlabel('Epoch (10 episodes)')
    plt.ylabel('Average Score')
    plt.axhline(y=-100, color='r', linestyle='--', label='Solving threshold (-100)')
    plt.legend()
    plt.savefig('figures/dqn_vs_dueling_dqn_comparison.png')
    plt.show()

    env.close()


if __name__ == '__main__':
    main()
