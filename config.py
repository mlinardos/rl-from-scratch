from dataclasses import dataclass
import random
import numpy as np
import torch


@dataclass
class Config:
    seed: int = 100
    env_id: str = 'Acrobot-v1'
    replay_memory_size: int = 10000
    batch_size: int = 32
    gamma: float = 0.99
    target_update_freq: int = 100
    learning_rate: float = 0.00025
    rms_decay: float = 0.95
    rms_momentum: float = 0.95
    rms_epsilon: float = 0.01
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 2000
    sequence_length: int = 4  # for GRU-DQN


def setup(config: Config) -> torch.device:
    """Set random seeds and return the compute device."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device
