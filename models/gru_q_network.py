import torch.nn as nn


class GRUQNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size=state_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])
