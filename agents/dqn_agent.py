import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import QNetwork, DuelingQNetwork
from memory import ReplayMemory, Experience
from config import Config


class DQNAgent:
    def __init__(self, state_size: int, action_size: int, device: torch.device, config: Config, network_type: str = "standard"):
        if network_type == "standard":
            self.policy_net = QNetwork(state_size, action_size).to(device)
            self.target_net = QNetwork(state_size, action_size).to(device)
        elif network_type == "dueling":
            self.policy_net = DuelingQNetwork(state_size, action_size).to(device)
            self.target_net = DuelingQNetwork(state_size, action_size).to(device)
        else:
            raise ValueError("network_type must be 'standard' or 'dueling'")

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(),
            lr=config.learning_rate,
            alpha=config.rms_decay,
            momentum=config.rms_momentum,
            eps=config.rms_epsilon,
            centered=False,
        )

        self.memory = ReplayMemory(config.replay_memory_size)
        self.config = config
        self.device = device
        self.action_size = action_size
        self.step_count = 0
        self.network_type = network_type

    def select_action(self, state, epsilon: float) -> int:
        if random.random() > epsilon:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.policy_net(state_t).max(1)[1].item()
        return random.randrange(self.action_size)

    def optimize_model(self):
        cfg = self.config
        if len(self.memory) < cfg.batch_size:
            return

        experiences = self.memory.sample(cfg.batch_size)
        batch = Experience(*zip(*experiences))

        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(np.array(batch.action)).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(np.array(batch.done)).unsqueeze(1).to(self.device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)

        expected_state_action_values = torch.where(
            done_batch == 1,
            reward_batch,
            reward_batch + cfg.gamma * next_state_values,
        )

        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
