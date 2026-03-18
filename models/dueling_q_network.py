import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)

        self.value_fc = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1)

        self.advantage_fc = nn.Linear(128, 128)
        self.advantage = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))

        value = torch.relu(self.value_fc(x))
        value = self.value(value)

        advantage = torch.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)

        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return value + advantage - advantage.mean(dim=1, keepdim=True)
