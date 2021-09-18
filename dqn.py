from torch import nn


class DQN(nn.Module):
    """Simple MLP network.
    >>> DQN(10, 5)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    DQN(
      (net): Sequential(...)
    )
    """

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 512):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        print(obs_size)
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, n_actions))

    def forward(self, x):
        return self.net(x.float())
