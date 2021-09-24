import torch
from torch import nn
from battler.utils.pokemon_embed import PokemonEmbed, MoveEmbed


class DQN(nn.Module):
    """Simple MLP network.
    >>> DQN(10, 5)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    DQN(
      (net): Sequential(...)
    )
    """

    def __init__(
        self, 
        obs_size: int = 1, 
        stack_size: int = 1,
        n_actions: int = 1, 
        out_channels: int = 1,
        hidden_size: int = 512,
        num_layers: int = 2,
    ):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()

        self.obs_size = obs_size
        self.stack_size = stack_size
        self.n_actions = n_actions
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Required for tensorboard graph from lightning
        self.example_input_array = torch.zeros(stack_size, obs_size) 

        #layers += [
            # 4 x 184 -> 184
            #nn.Conv1d(stack_size, 1, kernel_size=obs_size),
            #nn.ReLU(),
            #nn.Flatten(),
        #]

        print(f"obs_size: {obs_size}")
        print(f"stack_size: {stack_size}")
        print(f"n_actions: {n_actions}")
        #self.conv = nn.Conv1d(obs_size, 1, kernel_size=stack_size)
        # TODO could just remember n dims are this value or that values

        layers = []
        layers += [
            #self.conv
            # [batch_size, stack_size, obs_size]
            nn.Conv1d(stack_size, out_channels, kernel_size=1),
            nn.Flatten(),
        ]

        layers += [
            #nn.Linear(obs_size*stack_size, hidden_size), 
            nn.Linear(obs_size*out_channels, hidden_size), 
            #nn.Linear(obs_size*stack_size, hidden_size), 
            nn.ReLU(), 
        ]

        # Hidden
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        #layers.append(nn.Linear(hidden_size, n_actions))

        # Action values
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

        self.net = nn.Sequential(*layers)
    
    def get_kwargs(self):
        return {
            "obs_size": self.obs_size,
            "stack_size": self.stack_size,
            "n_actions": self.n_actions,
            "out_channels": self.out_channels,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
        }

    def forward(self, x):
        #pls = self.conv(x)
        #print(f"post_conv: {pls.shape}")
        #x = nn.functional.flatten(x, dim=1)
        #print(f"x size: {x.size()}")

        features = self.net(x.float())
        values = self.value(features)
        advantages = self.advantage(features)
        qvals = values + (advantages - advantages.mean())
        return qvals
