import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import GRUQNetwork
from memory import TrajectoryReplayMemory
from config import Config


class GRUDQNAgent:
    def __init__(self, state_size: int, action_size: int, device: torch.device, config: Config):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.config = config

        self.policy_net = GRUQNetwork(state_size, action_size).to(device)
        self.target_net = GRUQNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(),
            lr=config.learning_rate,
            alpha=config.rms_decay,
            eps=config.rms_epsilon,
            momentum=config.rms_momentum,
            centered=False,
        )

        self.memory = TrajectoryReplayMemory(config.replay_memory_size, config.sequence_length)
        self.state_buffer = deque(maxlen=config.sequence_length)
        self.reset_state_buffer()
        self.steps_done = 0

    def reset_state_buffer(self):
        self.state_buffer = deque(maxlen=self.config.sequence_length)
        for _ in range(self.config.sequence_length):
            self.state_buffer.append(np.zeros(self.state_size))

    def update_state_buffer(self, state):
        self.state_buffer.append(state)

    def select_action(self, epsilon: float) -> int:
        if random.random() > epsilon:
            states = np.array(list(self.state_buffer))
            states_t = torch.FloatTensor(states).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.policy_net(states_t).max(1)[1].item()
        return random.randrange(self.action_size)

    def optimize_model(self):
        cfg = self.config
        if len(self.memory) < cfg.batch_size:
            return

        experiences = self.memory.sample(cfg.batch_size)

        states_batch = torch.FloatTensor(np.array([e.states for e in experiences])).to(self.device)
        actions_batch = torch.LongTensor(np.array([e.actions[-1] for e in experiences])).unsqueeze(1).to(self.device)
        rewards_batch = torch.FloatTensor(np.array([e.rewards[-1] for e in experiences])).unsqueeze(1).to(self.device)
        next_states_batch = torch.FloatTensor(np.array([e.next_states for e in experiences])).to(self.device)
        dones_batch = torch.FloatTensor(np.array([e.dones[-1] for e in experiences])).unsqueeze(1).to(self.device)

        state_action_values = self.policy_net(states_batch).gather(1, actions_batch)

        with torch.no_grad():
            next_state_values = self.target_net(next_states_batch).max(1)[0].unsqueeze(1)

        expected = rewards_batch + (1 - dones_batch) * cfg.gamma * next_state_values

        loss = nn.SmoothL1Loss()(state_action_values, expected)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
