from collections import deque, namedtuple
import random
import numpy as np

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
GRUExperience = namedtuple('GRUExperience', field_names=['states', 'actions', 'rewards', 'next_states', 'dones'])


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class TrajectoryReplayMemory:
    def __init__(self, capacity: int, sequence_length: int = 4):
        self.memory = deque(maxlen=capacity)
        self.sequence_length = sequence_length
        self._reset_buffers()

    def _reset_buffers(self):
        self.current_states = []
        self.current_actions = []
        self.current_rewards = []
        self.current_next_states = []
        self.current_dones = []

    def push(self, state, action, reward, next_state, done):
        self.current_states.append(state)
        self.current_actions.append(action)
        self.current_rewards.append(reward)
        self.current_next_states.append(next_state)
        self.current_dones.append(done)

        if len(self.current_states) == self.sequence_length or done:
            if len(self.current_states) >= 2:
                while len(self.current_states) < self.sequence_length:
                    self.current_states.append(np.zeros_like(state))
                    self.current_actions.append(0)
                    self.current_rewards.append(0)
                    self.current_next_states.append(np.zeros_like(next_state))
                    self.current_dones.append(True)

                self.memory.append(GRUExperience(
                    states=np.array(self.current_states),
                    actions=np.array(self.current_actions),
                    rewards=np.array(self.current_rewards),
                    next_states=np.array(self.current_next_states),
                    dones=np.array(self.current_dones),
                ))

            self._reset_buffers()

    def sample(self, batch_size: int):
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def __len__(self):
        return len(self.memory)
