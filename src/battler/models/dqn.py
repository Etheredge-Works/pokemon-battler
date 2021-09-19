from torch import nn


class DQN(nn.Module):
    """Simple MLP network.
    >>> DQN(10, 5)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    DQN(
      (net): Sequential(...)
    )
    """

    def __init__(
        self, 
        obs_size: int, 
        n_actions: int, 
        hidden_size: int = 512,
        num_layers: int = 2
    ):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """

        self.action_size = n_actions
        super().__init__()
        print(obs_size)
        layers = [
            nn.Linear(obs_size, hidden_size), 
            nn.ReLU(), 
        ]

        # Hidden
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, n_actions))

        # Action values
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())
