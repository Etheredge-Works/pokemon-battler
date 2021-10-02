# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Lightning implementation of Proximal Policy Optimization (PPO)
<https://arxiv.org/abs/1707.06347>
Paper authors: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
The example implements PPO compatible to work with any continous or discrete action-space environments via OpenAI Gym.
To run the template, just run:
`python reinforce_learn_ppo.py`
References
----------
[1] https://github.com/openai/baselines/blob/master/baselines/ppo2/ppo2.py
[2] https://github.com/openai/spinningup
[3] https://github.com/sid-sundrani/ppo_lightning
"""
import argparse
from typing import Callable, Iterator, List, Tuple

import gym
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, IterableDataset
import torch.nn.functional as F
from torchsummary import summary
from contextlib import contextmanager, redirect_stdout

import pytorch_lightning as pl
from pl_examples import cli_lightning_logo
from copy import deepcopy
from icecream import ic
from battler.utils.encoders.abilities import GEN8_POKEMON, GEN8_ABILITIES
import random
from collections import deque


from poke_env.environment.weather import Weather
from poke_env.environment.status import Status
from poke_env.environment.pokemon_type import PokemonType

class PokemonNet(nn.Module):
    def __init__(self,
        name_embedding_dim: int = 10,
        status_embedding_dim: int = 2,
        type_embedding_dim: int = 3,
        ability_embedding_dim: int = 6,
        item_embedding_dim: int = 10,
    ):
        super().__init__()
        self.name_embedding = nn.Embedding(len(GEN8_POKEMON)+2, 10)
        self.status_embedding = nn.Embedding(len(Status)+2, 2)

        # used twice
        self.type1_embedding = nn.Embedding(len(PokemonType)+2, 3) # 1 higher for None
        self.type2_embedding = self.type1_embedding
        #_type2_embedding = nn.Embedding(len(PokemonType)+2, 3) # 1 higher for None

        self.ability_embedding = nn.Embedding(272, 6)
        # TODO verify dims
        self.item_embedding = nn.Embedding(923, 10)

        self.per_poke_enum_dim = \
            self.name_embedding.embedding_dim + \
            self.type1_embedding.embedding_dim + \
            self.type2_embedding.embedding_dim + \
            self.ability_embedding.embedding_dim + \
            self.item_embedding.embedding_dim + \
            self.status_embedding.embedding_dim

        self.net = nn.Sequential(
            #nn.Conv1d(poke_embedding_dim, hidden_size, 1), # TODO convert to conv
            nn.Linear(self.per_poke_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
    
    def input_dim(self):
        return 126

    def output_dim(self):
        return 64

    def forward(self, x):
        x_ints = x[:, :, :36].long()

        names = self.name_embedding(x_ints[:, :, 0:6]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        status = self.status_embedding(x_ints[:, :, 6:12]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        type1 = self.type1_embedding(x_ints[:, :, 12:18]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        type2 = self.type2_embedding(x_ints[:, :, 18:24]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        ability = self.ability_embedding(x_ints[:, :, 24:30]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        item = self.item_embedding(x_ints[:, :, 30:36]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)

        pokes = x[:, :, 36:126].view(x_ints.shape[0], x_ints.shape[1], 6, 15)

        poke_x = torch.cat([pokes, names, status, type1, type2, ability, item], dim=-1)

        poke_x = self.per_poke_net(poke_x)
        poke_x = poke_x.view(x_ints.shape[0], x_ints.shape[1], -1)
        return poke_x


from dataclasses import dataclass


# TODO move to passing in embeddings
@dataclass
class Embeddings:
    weather_embedding = nn.Embedding((len(Weather)+2), 2)
    name_embedding = nn.Embedding(len(GEN8_POKEMON)+2, 10)
    status_embedding = nn.Embedding(len(Status)+2, 2)
    # used twice
    type_embedding = nn.Embedding(len(PokemonType)+2, 3) # 1 higher for None
    #_type2_embedding = nn.Embedding(len(PokemonType)+2, 3) # 1 higher for None
    ability_embedding = nn.Embedding(272, 6)
    # TODO verify dims
    item_embedding = nn.Embedding(923, 10)


class PokeMLP(nn.Module):
    #NUM_TYPES = ##6
    '''
    layout:
    0: weather
    1-6: pokemons_names
    7-12: pokemons status
    13-18: pokemons types 1
    19-24: pokemons types 2
    25-30: pokemons abilities
    31-36: pokemons items
    31-36: pokemons items

    '''
    _weather_embedding = nn.Embedding((len(Weather)+2), 4)
    ic(_weather_embedding)
    #ic(sum(_weather_embedding.parameters()))
    _name_embedding = nn.Embedding(len(GEN8_POKEMON)+2, 16)
    #ic(_name_embedding.parameters().numel())
    _status_embedding = nn.Embedding(len(Status)+2, 8)
    #ic(_status_embedding.parameters().numel())
    # used twice
    _type1_embedding = nn.Embedding(len(PokemonType)+2, 12) # 1 higher for None
    _type2_embedding = _type1_embedding
    #_type2_embedding = nn.Embedding(len(PokemonType)+2, 3) # 1 higher for None
    _ability_embedding = nn.Embedding(272, 16)
    # TODO verify dims
    _item_embedding = nn.Embedding(923, 16)

    # TODO use layer input size here
    per_poke_enum_dim = \
        _name_embedding.embedding_dim + \
        _type1_embedding.embedding_dim + \
        _type2_embedding.embedding_dim + \
        _ability_embedding.embedding_dim + \
        _item_embedding.embedding_dim + \
        _status_embedding.embedding_dim
    ic(per_poke_enum_dim)
        #10 + 2 + 3 + 3 + 6 + 10
    #ic(per_poke_enum_dim)

    # Net for pokemon info
    poke_floats = 15
    per_poke_embedding_dim = per_poke_enum_dim + poke_floats
    #ic(per_poke_embedding_dim)
    move_dim = 20
    '''
    _per_poke_net = nn.Sequential(
        #nn.Conv1d(poke_embedding_dim, hidden_size, 1), # TODO convert to conv
        #nn.LazyLinear(per_poke_embedding_dim, 256),
        nn.Linear(per_poke_embedding_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        #nn.ReLU(),
    )

    #_opp_per_poke_net = deepcopy(_per_poke_net)
    _opp_per_poke_net = _per_poke_net
    # TODO test opp_poke_embedding_net = deepcopy(poke_embedding_net)

    _move_net = nn.Sequential(
        nn.Linear((move_dim-1) + _type1_embedding.embedding_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
    )
    '''


    def __init__(
        self, 
        input_shape: Tuple[int], 
        n_actions: int, 
        hidden_dim: int = 1024,
        num_dense_layers: int = 4,
    ):
        super().__init__()
        # I'm not sure why +2. +1 is to account for None
        self.weather_embedding = self._weather_embedding
        self.name_embedding = self._name_embedding
        self.status_embedding = self._status_embedding
        self.type1_embedding = self._type1_embedding
        self.type2_embedding = self._type2_embedding
        self.ability_embedding = self._ability_embedding
        self.item_embedding = self._item_embedding
        '''
        self.per_poke_net = nn.Sequential(
            #nn.Conv1d(poke_embedding_dim, hidden_size, 1), # TODO convert to conv
            #nn.LazyLinear(256),
            nn.Linear(self.per_poke_embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            #nn.ReLU(),
        )
        
        self.opp_poke_net = deepcopy(self.per_poke_net)
        ic(self.per_poke_net)
        #@self.opp_poke_net = deepcopy(self._opp_per_poke_net)
        #self.opp_poke_net = self.per_poke_net


        #self.move_net = deepcopy(self._move_net)
        self.move_net = nn.Sequential(
            nn.Linear((self.move_dim-1) + self._type1_embedding.embedding_dim, 64),
            #nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        #self.opp_move_net = deepcopy(self._move_net)
        self.opp_move_net = deepcopy(self.move_net)
        ic(self.move_net)

        # TODO param poke_net
        pre_converted_dims = 1 + ((15+6) * 12)
        ic(pre_converted_dims)
        #converted_dims = (self.per_poke_enum_dim * 12) + self.weather_embedding.embedding_dim
        post_converted_dims = (12 * 64) + self.weather_embedding.embedding_dim
        ic(post_converted_dims)

        #poke_dims = self.per_poke_net.
        embedding_dim_delta =  post_converted_dims - pre_converted_dims
        ic(embedding_dim_delta)
        conv_out_channels = 4

        #ic(input_shape)
        moves_flattened_dim = 6 * 4 * 8 * 2
        moves_dim_delta = moves_flattened_dim - (self.move_dim*12*4) - 1
        frame_dim = input_shape[1] + embedding_dim_delta + moves_dim_delta
        frame_dim = 2515
        net_input_dim = (input_shape[1] + embedding_dim_delta) * conv_out_channels
        net_input_dim = 2515 * 2
        ic(net_input_dim) # shoudl be 576
        '''
        # TODO keep testing reflex agents
        # TODO the agent does seem to behave differently based on previous attacks
        # TODO active poke net and other poke net
        total_dim = 1
        for dim in input_shape[0:]:
            total_dim *= dim

        layers = []
        layers.append(nn.Conv1d(input_shape[0], 64, 1))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv1d(64, 64, 1))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(total_dim, total_dim//3)) # filter down to half per frame
        #layers.append(nn.Linear(total_dim, total_dim//2)) # filter down to half per frame
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv1d(32, 4, 1))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Flatten(1))
        layers.append(nn.Linear((total_dim//3)*4, hidden_dim))
        #layers.append(nn.Linear(total_dim, hidden_dim))
        layers.append(nn.LeakyReLU())
        for _ in range(num_dense_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            #layers.append(nn.LazyLinear(hidden_dim))
            layers.append(nn.LeakyReLU())
        #layers.append(nn.LazyLinear(hidden_dim, n_actions))
        layers.append(nn.Linear(hidden_dim, n_actions))
        self.net = nn.Sequential(*layers)


        #self.net = nn.Sequential(
            #nn.Conv1d(input_shape[0], 64, kernel_size=1),
            #nn.BatchNorm1d(32),
            #nn.LazyLinear(hidden_dim),
            #nn.LeakyReLU(),
            #nn.LazyLinear(hidden_dim),
            #nn.LeakyReLU(),
            #nn.LazyLinear(hidden_dim),
            #nn.LeakyReLU(),
            #nn.LazyLinear(hidden_dim),
            #nn.LeakyReLU(),
            #nn.LazyLinear(hidden_dim),
            #nn.LeakyReLU(),
            # Shuffle inforation across frame
            #nn.Conv1d(input_shape[0], 64, kernel_size=1),
            #nn.BatchNorm1d(32),
            #nn.ReLU(),
            #nn.ReLU(),
            #nn.Conv1d(64, 64, kernel_size=1),
            #nn.BatchNorm1d(64),
            #nn.ReLU(),
            #nn.Linear(3103, 512),
            #nn.LazyLinear(1024),
            #nn.Conv1d(64, 64, kernel_size=1),
            #nn.BatchNorm1d(64),
            #nn.ReLU(),
            #nn.Linear(512, 256),
            #nn.Linear(frame_dim, frame_dim//2),
            #nn.LazyLinear(512),
            #nn.ReLU(),

            # Shuffle inforation inside frame
            #nn.Linear(frame_dim, frame_dim//2),
            #nn.Linear(hidden_size, hidden_size),
            #nn.ReLU(),

            #nn.Conv1d(64, 64, kernel_size=1),
            # TODO make sure using eval mode other places if using batchnorm? not used elsewhere
            #nn.BatchNorm1d(64),
            #nn.ReLU(),
            #nn.AdaptiveAvgPool1d(64),
            #nn.Flatten(),
            #nn.Linear(net_input_dim, hidden_size),
            #nn.Linear(hidden_size*conv_out_channels, hidden_size),
            #nn.Linear(hidden_size*conv_out_channels, hidden_size),
            #nn.LazyLinear(hidden_size),
            #nn.Dropout(0.2),
            #nn.Linear(net_input_dim//conv_out_channels, hidden_size),
            #nn.ReLU(),
            #nn.Linear(hidden_size, hidden_size),
            #nn.LazyLinear(hidden_size),
            #nn.ReLU(),
            #nn.LazyLinear(n_actions),
        #)
        ic(self.net)
        self.moves_range = self.move_dim * 4 * 6 

        self.per_poke_net = None
        self.opp_poke_net = None

        self.opp_move_net = None
        self.move_net = None
        self.opp_move_net = None

        #net_input_dim = (input_shape[0] * input_shape[1])
        #net_input_dim = 10


    def move_forward(self, x, net):
        moves_raw = x[:, : , :self.moves_range].view(*x.shape[0:2], 6, 4, self.move_dim)
        ##ic(moves_raw.shape)
        int_in = moves_raw[:, :, :, :, 0].long()
        #ic(int_in.shape)
        type_x = self.type1_embedding(int_in)
        #ic(type_x.shape)
        x = torch.cat((moves_raw[:, :, :, :, 1:], type_x), dim=-1)
        #ic(x.shape)
        return x
        y = net(x)
        return y
        #ic(y.shape)


    def poke_forward(self, x, net):
        x_ints = x[:, :, :36].long()

        names = self.name_embedding(x_ints[:, :, 0:6]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        status = self.status_embedding(x_ints[:, :, 6:12]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        type1 = self.type1_embedding(x_ints[:, :, 12:18]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        type2 = self.type2_embedding(x_ints[:, :, 18:24]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        ability = self.ability_embedding(x_ints[:, :, 24:30]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        item = self.item_embedding(x_ints[:, :, 30:36]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)

        pokes = x[:, :, 36:126].view(x_ints.shape[0], x_ints.shape[1], 6, 15)

        poke_x = torch.cat([pokes, names, status, type1, type2, ability, item], dim=-1)

        #poke_x = net(poke_x)
        poke_x = poke_x.view(x_ints.shape[0], x_ints.shape[1], -1)
        return poke_x
    
    def forward(self, x):
        if len(x.shape) == 4:
            # Needed because it one time gets 512x1x.x.
            x.squeeze_(1)
        elif len(x.shape) < 3:
            x.unsqueeze_(0)
        #ic(x.shape)

        #weather_data = x[:, :, 0]
        #ic(weather_data.shape)
        #ally_pokemon_data = x[:, :, 1:127].view(x.shape[0], x.shape[1], 6, 21)
        #ic(ally_pokemon_data.shape)


        #weather = self.weather_embedding(x[:, :, 0:1])
        #type_idx = [
            #*list(range(13, 25)), 
            #*list(range(12+127, 24+127)),
            #254, 274, 294, 114, 
        #]
        #ic(type_idx)



        # better way below ######

        #ic()
        #ic(x.shape)
        #print(f"input: {x.shape}")

        x_ints = torch.round(x[:, :, :1]).long()

        weather = self.weather_embedding(x_ints[:, :, 0:1]).view(x_ints.shape[0], x_ints.shape[1], -1)


        ally_moves_delta = 254 + self.moves_range
        #ic(ally_moves_delta)
        ally_move_x = self.move_forward(x[:, :, 254:ally_moves_delta], self.move_net)
        #ic(ally_move_x.shape)
        ally_move_x = ally_move_x.view(x_ints.shape[0], x_ints.shape[1], -1)

        opp_move_delts = ally_moves_delta + self.moves_range
        opp_move_x = self.move_forward(x[:, :, ally_moves_delta:opp_move_delts], self.opp_move_net)
        opp_move_x = self.move_forward(x[:, :, ally_moves_delta:opp_move_delts], self.opp_move_net)
        #ic(opp_move_x.shape)
        opp_move_x = opp_move_x.view(x_ints.shape[0], x_ints.shape[1], -1)
        #ic(opp_move_x.shape)

        # Friendly Pokes
        poke_x = self.poke_forward(x[:, :, 1:127], self.per_poke_net)

        # Opponent Pokes
        opp_poke_x = self.poke_forward(x[:, :, 127:254], self.opp_poke_net)

        # TODO handle type encoding in moves

        x = torch.cat((poke_x, opp_poke_x, weather, ally_move_x, opp_move_x, x[:, :, opp_move_delts:]), dim=-1)
        #ic(x.shape)

        x = self.net(x)
        
        return x.squeeze(0)
    

class ActorCategorical(nn.Module):
    """Policy network, for discrete action spaces, which returns a distribution and an action given an
    observation."""

    def __init__(self, actor_net):
        """
        Args:
            input_shape: observation shape of the environment
            n_actions: number of discrete actions available in the environment
        """
        super().__init__()

        self.actor_net = actor_net

    def forward(self, states):
        logits = self.actor_net(states)
        pi = Categorical(logits=logits)
        actions = pi.sample()

        return pi, actions

    def get_log_prob(self, pi: Categorical, actions: torch.Tensor):
        """Takes in a distribution and actions and returns log prob of actions under the distribution.
        Args:
            pi: torch distribution
            actions: actions taken by distribution
        Returns:
            log probability of the acition under pi
            # TODO update on pytroch lightning docs
        """
        return pi.log_prob(actions)


class ActorContinous(nn.Module):
    """Policy network, for continous action spaces, which returns a distribution and an action given an
    observation."""

    def __init__(self, actor_net, act_dim):
        """
        Args:
            input_shape: observation shape of the environment
            n_actions: number of discrete actions available in the environment
        """
        super().__init__()
        self.actor_net = actor_net
        log_std = -0.5 * torch.ones(act_dim, dtype=torch.float)
        self.log_std = nn.Parameter(log_std)

    def forward(self, states):
        mu = self.actor_net(states)
        std = torch.exp(self.log_std)
        pi = Normal(loc=mu, scale=std)
        actions = pi.sample()

        return pi, actions

    def get_log_prob(self, pi: Normal, actions: torch.Tensor):
        """Takes in a distribution and actions and returns log prob of actions under the distribution.
        Args:
            pi: torch distribution
            actions: actions taken by distribution
        Returns:
            log probability of the acition under pi
        """
        return pi.log_prob(actions).sum(axis=-1)


class ExperienceSourceDataset(IterableDataset):
    """Implementation from PyTorch Lightning Bolts: https://github.com/PyTorchLightning/lightning-
    bolts/blob/master/pl_bolts/datamodules/experience_source.py.
    Basic experience source dataset. Takes a generate_batch function that returns an iterator. The logic for the
    experience source and how the batch is generated is defined the Lightning model itself
    """

    def __init__(self, generate_batch: Callable):
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterator:
        iterator = self.generate_batch()
        return iterator


class PPOLightning(pl.LightningModule):
    """PyTorch Lightning implementation of PPO.
    Example:
        model = PPOLightning("CartPole-v0")
    Train:
        trainer = Trainer()
        trainer.fit(model)
    """

    def __init__(
        self,
        env_name: str,
        env_kwargs: dict = None,
        net_kwargs: dict = None,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        max_episode_len: float = 100,
        batch_size: int = 512,
        #batch_size: int = 1024,
        steps_per_epoch: int = 4096, # increased due to randomness of pokes
        #steps_per_epoch: int = 8192, # increased due to randomness of pokes
        nb_optim_iters: int = 4,
        clip_ratio: float = 0.2,
        n_policies: int = 100,
        policy_update_threshold: float = 0.6,
        **kwargs,
    ) -> None:
        """
        Args:
            env: gym environment tag
            gamma: discount factor
            lam: advantage discount factor (lambda in the paper)
            lr_actor: learning rate of actor network
            lr_critic: learning rate of critic network
            max_episode_len: maximum number interactions (actions) in an episode
            batch_size:  batch_size when training network- can simulate number of policy updates performed per epoch
            steps_per_epoch: how many action-state pairs to rollout for trajectory collection per epoch
            nb_optim_iters: how many steps of gradient descent to perform on each batch
            clip_ratio: hyperparameter for clipping in the policy objective
        """
        super().__init__(**kwargs)
        #env_kwargs = env_kwargs or {}

        # TODO separate networks for active pokemon? is hat not already what it's doing?
        env_kwargs = env_kwargs if env_kwargs is not None else {}
        net_kwargs = net_kwargs if net_kwargs is not None else {}

        # Hyperparameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.steps_per_epoch = steps_per_epoch
        self.nb_optim_iters = nb_optim_iters
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.max_episode_len = max_episode_len
        self.clip_ratio = clip_ratio
        self.n_policies = n_policies
        self.policy_update_threshold = policy_update_threshold
        self.env = gym.make(env_name, **env_kwargs)

        self.save_hyperparameters()

        self.player_id = self.env.username
        self.save_hyperparameters(dict(player_id=self.player_id))

        # value network
        self.critic = PokeMLP(self.env.observation_space.shape, 1, **net_kwargs)

        # policy network (agent)
        if isinstance(self.env.action_space, gym.spaces.box.Box):
            act_dim = self.env.action_space.shape[0]
            actor_mlp = PokeMLP(self.env.observation_space.shape, act_dim, **net_kwargs)
            #actor_mlp = create_mlp(self.env.observation_space.shape, act_dim)
            self.actor = ActorContinous(actor_mlp, act_dim)
        elif isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            actor_mlp = PokeMLP(self.env.observation_space.shape, self.env.action_space.n, **net_kwargs)
            #actor_mlp = create_mlp(self.env.observation_space.shape, self.env.action_space.n)
            self.actor = ActorCategorical(actor_mlp)
        else:
            raise NotImplementedError(
                "Env action space should be of type Box (continous) or Discrete (categorical)."
                f" Got type: {type(self.env.action_space)}"
            )
        
        self.batch_states = []
        self.batch_actions = []
        self.batch_adv = []
        self.batch_qvals = []
        self.batch_logp = []

        self.ep_rewards = []
        self.ep_values = []
        self.epoch_rewards = []

        self.episode_step = 0
        self.avg_ep_reward = 0
        self.avg_ep_len = 0
        self.avg_reward = 0

        self.games_played = 0
        self.n_wins = 0
        self.n_battles = 0
        self.best_win_rate = 0

        #state_ints, state_floats = self.env.reset()
        #self.state = torch.FloatTensor(state_floats), torch.IntTensor(state_ints)
        self.state = torch.FloatTensor(self.env.reset())

        self.actor_bank = deque(maxlen=self.n_policies)
        self.n_policies_pushed = 0
        #self.policy_bank.append(self.build_policy(self.actor))

   
    def on_train_start(self) -> None:
        critic_path = 'critic_summary.txt'
        with open(critic_path, 'w') as f:
            with redirect_stdout(f):
                print(self.critic)
                #summary(self.critic, self.env.observation_space.shape)
        self.logger.experiment.log_artifact(self.logger.run_id, local_path=critic_path)

        actor_path = 'actor_summary.txt'
        with open(actor_path, 'w') as f:
            with redirect_stdout(f):
                print(self.actor)
                #summary(self.actor, self.env.observation_space.shape)
        self.logger.experiment.log_artifact(self.logger.run_id, local_path=actor_path)

        return super().on_train_start()


    def log_actor(self):
        #local_path = f"{self.logger.save_dir}/actor.pt"
        local_path = f"actor.pt"
        torch.save(self.actor.state_dict(), local_path)
        #actor_kwargs = self.net_kwargs.copy()
        self.logger.experiment.log_artifact(self.logger.run_id, local_path=local_path)

        #mlflow.pytorch.log_model(self.actor, "actor", run_id=self.logger.run_id)
        # will this work? mlflow.pytorch.log_model(self.actor, "actor", run_id=self.logger.run_id)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Passes in a state x through the network and returns the policy and a sampled action.
        Args:
            x: environment state
        Returns:
            Tuple of policy and action
        """
        pi, action = self.actor(x)
        value = self.critic(x)

        return pi, action, value

    def discount_rewards(self, rewards: List[float], discount: float) -> List[float]:
        """Calculate the discounted rewards of all rewards in list.
        Args:
            rewards: list of rewards/advantages
        Returns:
            list of discounted rewards/advantages
        """
        assert isinstance(rewards[0], float)

        cumul_reward = []
        sum_r = 0.0

        for r in reversed(rewards):
            sum_r = (sum_r * discount) + r
            cumul_reward.append(sum_r)

        return list(reversed(cumul_reward))

    def calc_advantage(self, rewards: List[float], values: List[float], last_value: float) -> List[float]:
        """Calculate the advantage given rewards, state values, and the last value of episode.
        Args:
            rewards: list of episode rewards
            values: list of state values from critic
            last_value: value of last state of episode
        Returns:
            list of advantages
        """
        rews = rewards + [last_value]
        vals = values + [last_value]
        # GAE
        delta = [rews[i] + self.gamma * vals[i + 1] - vals[i] for i in range(len(rews) - 1)]
        adv = self.discount_rewards(delta, self.gamma * self.lam)

        return adv

    def generate_trajectory_samples(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Contains the logic for generating trajectory data to train policy and value network
        Yield:
           Tuple of Lists containing tensors for states, actions, log probs, qvals and advantage
        """

        for step in range(self.steps_per_epoch):
            self.state = self.state.to(device=self.device)

            with torch.no_grad():
                pi, action, value = self(self.state)
                log_prob = self.actor.get_log_prob(pi, action)

            next_state, reward, done, _ = self.env.step(action.cpu().numpy())

            self.episode_step += 1

            self.batch_states.append(self.state)
            self.batch_actions.append(action)
            self.batch_logp.append(log_prob)

            self.ep_rewards.append(reward)
            self.ep_values.append(value.item())

            #self.state = torch.IntTensor(state_ints), torch.FloatTensor(state_floats)
            self.state = torch.FloatTensor(next_state)

            epoch_end = step == (self.steps_per_epoch - 1)
            terminal = len(self.ep_rewards) == self.max_episode_len

            if epoch_end or done or terminal:
                self.games_played += 1
                # if trajectory ends abtruptly, boostrap value of next state
                if (terminal or epoch_end) and not done:
                    self.state = self.state.to(device=self.device)
                    with torch.no_grad():
                        _, _, value = self(self.state)
                        last_value = value.item()
                        steps_before_cutoff = self.episode_step
                else:
                    last_value = 0
                    steps_before_cutoff = 0

                # discounted cumulative reward
                self.batch_qvals += self.discount_rewards(self.ep_rewards + [last_value], self.gamma)[:-1]
                # advantage
                self.batch_adv += self.calc_advantage(self.ep_rewards, self.ep_values, last_value)
                # logs
                self.epoch_rewards.append(sum(self.ep_rewards))
                # reset params
                self.ep_rewards = []
                self.ep_values = []
                self.episode_step = 0
                self.update_opponent()
                self.state = torch.FloatTensor(self.env.reset())

            # TODO time
            if epoch_end:
                train_data = zip(
                    self.batch_states, self.batch_actions, self.batch_logp, self.batch_qvals, self.batch_adv
                )

                for state, action, logp_old, qval, adv in train_data:
                    yield state, action, logp_old, qval, adv

                self.batch_states.clear()
                self.batch_actions.clear()
                self.batch_adv.clear()
                self.batch_logp.clear()
                self.batch_qvals.clear()

                # logging
                self.avg_reward = sum(self.epoch_rewards) / self.steps_per_epoch

                # if epoch ended abruptly, exlude last cut-short episode to prevent stats skewness
                epoch_rewards = self.epoch_rewards
                if not done:
                    epoch_rewards = epoch_rewards[:-1]

                total_epoch_reward = sum(epoch_rewards)
                nb_episodes = len(epoch_rewards)

                self.avg_ep_reward = total_epoch_reward / nb_episodes
                self.avg_ep_len = (self.steps_per_epoch - steps_before_cutoff) / nb_episodes

                self.epoch_rewards.clear()

    def build_policy(self, net, device='cpu'):
        """
        Builds a policy from a given network
        Args:
            net: network to build policy from
        Returns:
            policy
        """

        net = deepcopy(net).to(device=self.device)
        net.eval()
        def policy(state):
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device=self.device)
                pi, action = net(state)
                return action.cpu().item()

        return policy

    def update_opponent(self):
        """
        Update the opponent's policy to match the current policy
        """
        # TODO should this be here?
        if len(self.actor_bank) > 0:
            actor = random.choice(self.actor_bank)
            self.env.set_opponent_policy(self.build_policy(actor))
    
    def on_epoch_end(self) -> None:
        super().on_epoch_end()


        wins_delta = self.env.n_won_battles - self.n_wins
        battles_delta = self.env.n_finished_battles - self.n_battles
        win_rate = wins_delta / battles_delta if battles_delta > 0 else 0

        self.log("n_battles", self.env.n_finished_battles)
        self.log("n_battles_per_epoch", battles_delta)
        self.log("win_rate", win_rate)

        self.n_wins = self.env.n_won_battles
        self.n_battles = self.env.n_finished_battles
        # TODO does this break algorithm?

        # TODO really should evaluate agent against old agent instead of this way
        # TODO but this way does make sure agent doesn't overfit and stays
        #      in the same performance relative to old policies
        self.log("n_old_actors", len(self.actor_bank))
        if win_rate > self.policy_update_threshold:
            self.actor_bank.append(deepcopy(self.actor).cpu())
            self.log_actor()

        if win_rate > self.best_win_rate:
            # not really useful
            self.best_win_rate = win_rate
            self.log("best_win_rate", self.best_win_rate)

            #mlflow.pytorch.log_model(self.actor, "actor", run_id=self.logger.run_id)
            #self.logger.experiment.
            #mlflow.pytorch.log_model(self.actor, 'actor', )


    def fight_max(self):
        def max_p(battle):
            if battle.available_actions:
                best_move = max(battle.available_actions, key=lambda a: a.base_power)
                return self.create_order(best_move)
        #self.env.set_opponent_policy(l())


    def actor_loss(self, state, action, logp_old, qval, adv) -> torch.Tensor:
        pi, _ = self.actor(state)
        logp = self.actor.get_log_prob(pi, action)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_actor = -(torch.min(ratio * adv, clip_adv)).mean()
        return loss_actor

    def critic_loss(self, state, action, logp_old, qval, adv) -> torch.Tensor:
        value = self.critic(state)
        loss_critic = (qval - value).pow(2).mean()
        return loss_critic

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx, optimizer_idx):
        """Carries out a single update to actor and critic network from a batch of replay buffer.
        Args:
            batch: batch of replay buffer/trajectory data
            batch_idx: not used
            optimizer_idx: idx that controls optimizing actor or critic network
        Returns:
            loss
        """
        state, action, old_logp, qval, adv = batch

        # normalize advantages
        adv = (adv - adv.mean()) / adv.std()

        self.log("avg_ep_len", self.avg_ep_len, prog_bar=True, on_step=False, on_epoch=True)
        self.log("avg_ep_reward", self.avg_ep_reward, prog_bar=True, on_step=False, on_epoch=True)
        self.log("avg_reward", self.avg_reward, prog_bar=False, on_step=False, on_epoch=True)

        if optimizer_idx == 0:
            loss_actor = self.actor_loss(state, action, old_logp, qval, adv)
            self.log("loss_actor", loss_actor, on_step=False, on_epoch=True, prog_bar=False, logger=True)

            return loss_actor

        if optimizer_idx == 1:
            loss_critic = self.critic_loss(state, action, old_logp, qval, adv)
            self.log("loss_critic", loss_critic, on_step=False, on_epoch=True, prog_bar=False, logger=True)

            return loss_critic

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        return optimizer_actor, optimizer_critic

    def optimizer_step(self, *args, **kwargs):
        """Run 'nb_optim_iters' number of iterations of gradient descent on actor and critic for each data
        sample."""
        for _ in range(self.nb_optim_iters):
            super().optimizer_step(*args, **kwargs)

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = ExperienceSourceDataset(self.generate_trajectory_samples)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self._dataloader()

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = parent_parser.add_argument_group("PPOLightning")
        parser.add_argument("--env", type=str, default="CartPole-v0")
        parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
        parser.add_argument("--lam", type=float, default=0.95, help="advantage discount factor")
        parser.add_argument("--lr_actor", type=float, default=3e-4, help="learning rate of actor network")
        parser.add_argument("--lr_critic", type=float, default=1e-3, help="learning rate of critic network")
        parser.add_argument("--max_episode_len", type=int, default=1000, help="capacity of the replay buffer")
        parser.add_argument("--batch_size", type=int, default=512, help="batch_size when training network")
        parser.add_argument(
            "--steps_per_epoch",
            type=int,
            default=2048,
            help="how many action-state pairs to rollout for trajectory collection per epoch",
        )
        parser.add_argument(
            "--nb_optim_iters", type=int, default=4, help="how many steps of gradient descent to perform on each batch"
        )
        parser.add_argument(
            "--clip_ratio", type=float, default=0.2, help="hyperparameter for clipping in the policy objective"
        )

        return parent_parser


def main(args) -> None:
    model = PPOLightning(**vars(args))

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model)


if __name__ == "__main__":
    cli_lightning_logo()
    pl.seed_everything(0)

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)

    parser = PPOLightning.add_model_specific_args(parent_parser)
    args = parser.parse_args()

    main(args)