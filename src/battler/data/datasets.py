from collections import deque, namedtuple
from typing import Tuple
import numpy as np
from torch.utils.data.dataset import Dataset, IterableDataset, Iterator

from .buffers import ReplayBuffer

class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.
    >>> RLDataset(ReplayBuffer(5))  # doctest: +ELLIPSIS
    <...reinforce_learn_Qnet.RLDataset object at ...>
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 10) -> None:
        """
        Args:
            buffer: replay buffer
            sample_size: number of experiences to sample at a time
        """
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]

