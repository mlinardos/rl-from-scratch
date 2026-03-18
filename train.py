import numpy as np
import gymnasium as gym

from config import Config
from agents.dqn_agent import DQNAgent
from agents.gru_dqn_agent import GRUDQNAgent


def train_dqn(env: gym.Env, agent: DQNAgent, config: Config, num_episodes: int = 500, max_steps: int = 500):
    scores = []
    total_steps = 0
    epoch_size = 10
    epoch_scores = []
    epsilon = config.epsilon_end

    label = agent.network_type.capitalize()
    print(f"Starting {label} DQN training for {config.env_id}")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            decay_progress = min(total_steps / config.epsilon_decay_steps, 1.0)
            epsilon = max(config.epsilon_end, config.epsilon_start - decay_progress * (config.epsilon_start - config.epsilon_end))

            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            total_steps += 1

            agent.optimize_model()

            if done:
                break

        scores.append(total_reward)

        if episode % epoch_size == 0:
            epoch_scores.append(np.mean(scores[-epoch_size:]))
            print(f"Epoch {len(epoch_scores)}, Average score: {epoch_scores[-1]:.2f}, Epsilon: {epsilon:.3f}, steps: {total_steps}")

        if episode % 100 == 0:
            print(f"Episode {episode}\tAverage Score (100 episodes): {np.mean(scores[-100:]):.2f}")

        if episode >= 100 and np.mean(scores[-100:]) >= -100.0:
            print(f"\nEnvironment solved in {episode} episodes. Average Score (100 episodes): {np.mean(scores[-100:]):.2f}")
            break

    return scores, epoch_scores


def train_gru_dqn(env: gym.Env, agent: GRUDQNAgent, config: Config, num_episodes: int = 500, max_steps: int = 500):
    scores = []
    total_steps = 0
    epoch_size = 10
    epoch_scores = []
    epsilon = config.epsilon_end

    print(f"Starting GRU DQN training for {config.env_id}")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0

        agent.reset_state_buffer()
        agent.update_state_buffer(state)

        for _ in range(max_steps):
            decay_progress = min(total_steps / config.epsilon_decay_steps, 1.0)
            epsilon = max(config.epsilon_end, config.epsilon_start - decay_progress * (config.epsilon_start - config.epsilon_end))

            action = agent.select_action(epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, done)
            agent.update_state_buffer(next_state)
            state = next_state
            total_reward += reward
            total_steps += 1

            agent.optimize_model()

            if done:
                break

        scores.append(total_reward)

        if episode % epoch_size == 0:
            epoch_scores.append(np.mean(scores[-epoch_size:]))
            print(f"Epoch {len(epoch_scores)}, Average score: {epoch_scores[-1]:.2f}, Epsilon: {epsilon:.3f}, steps: {total_steps}")

        if episode % 100 == 0:
            print(f"Episode {episode}\tAverage Score (100 episodes): {np.mean(scores[-100:]):.2f}")

        if episode >= 100 and np.mean(scores[-100:]) >= -100.0:
            print(f"\nEnvironment solved in {episode} episodes. Average Score (100 episodes): {np.mean(scores[-100:]):.2f}")
            break

    return scores, epoch_scores
