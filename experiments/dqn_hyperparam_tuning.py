import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import replace
import gymnasium as gym
import matplotlib.pyplot as plt

from config import Config, setup
from agents import DQNAgent
from train import train_dqn


def run_sweep(env, config, device, param_name, values, baseline_epochs, title, filename):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(baseline_epochs)), baseline_epochs,
             label=f"Baseline {param_name}={getattr(config, param_name)}", linewidth=2)

    for val in values:
        print(f"Running with {param_name} = {val}")
        cfg = replace(config, **{param_name: val})
        agent = DQNAgent(state_size, action_size, device, cfg)
        scores, epoch_scores = train_dqn(env, agent, cfg, num_episodes=500)
        plt.plot(range(len(epoch_scores)), epoch_scores, label=f"{param_name} = {val}")
        print(f"  Final avg score: {np.mean(scores[-100:]):.2f}")

    plt.title(title)
    plt.xlabel('Epoch (10 episodes)')
    plt.ylabel('Average Score per Epoch')
    plt.axhline(y=-100, color='r', linestyle='--')
    plt.legend()
    plt.savefig(filename)
    plt.show()


def main():
    config = Config()
    device = setup(config)

    env = gym.make(config.env_id)
    env.reset(seed=config.seed)
    env.action_space.seed(config.seed)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    print("--- Running baseline ---")
    baseline_agent = DQNAgent(state_size, action_size, device, config)
    _, baseline_epochs = train_dqn(env, baseline_agent, config, num_episodes=500)

    run_sweep(env, config, device,
              "target_update_freq", [50, 500, 1000], baseline_epochs,
              "Target Network Update Frequency Comparison",
              "acrobot_target_update_sensitivity.png")

    run_sweep(env, config, device,
              "replay_memory_size", [1000, 3000, 5000], baseline_epochs,
              "Replay Memory Size Comparison",
              "acrobot_memory_size_sensitivity.png")

    run_sweep(env, config, device,
              "epsilon_decay_steps", [500, 5000, 10000], baseline_epochs,
              "Epsilon Decay Steps Comparison",
              "acrobot_epsilon_decay_sensitivity.png")

    run_sweep(env, config, device,
              "learning_rate", [0.00001, 0.0001, 0.001], baseline_epochs,
              "Learning Rate Comparison",
              "acrobot_learning_rate_sensitivity.png")

    env.close()


if __name__ == '__main__':
    main()
