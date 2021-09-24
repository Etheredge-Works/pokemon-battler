
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
"""Deep Reinforcement Learning: Deep Q-network (DQN)
The template illustrates using Lightning for Reinforcement Learning. The example builds a basic DQN using the
classic CartPole environment.
To run the template, just run:
`python reinforce_learn_Qnet.py`
After ~1500 steps, you will see the total_reward hitting the max score of 475+.
Open up TensorBoard to see the metrics:
`tensorboard --logdir default`
References
----------
[1] https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-
Second-Edition/blob/master/Chapter06/02_dqn_pong.py
"""

import argparse
from copy import deepcopy
from collections import namedtuple, OrderedDict
from typing import List, Tuple, Dict

import gym
import numpy as np
from numpy.core.numeric import ones
from numpy.core.shape_base import stack
from poke_env.player.random_player import RandomPlayer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
import torch.nn.functional as F

import pytorch_lightning as pl
from pl_examples import cli_lightning_logo

from battler.players.opponents import MaxDamagePlayer, RandomOpponentPlayer

from ..data.datasets import ReplayBuffer, RLDataset
from ..data.buffers import Experience
from ..models.dqn import DQN
from .agents import Agent
from battler.utils.other import get_opponent
from battler.deploy.dqn_player import DQNPlayer
import asyncio


class DQNLightning(pl.LightningModule):
    """Basic DQN Model.
    >>> DQNLightning(env="CartPole-v1")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    DQNLightning(
      (net): DQN(
        (net): Sequential(...)
      )
      (target_net): DQN(
        (net): Sequential(...)
      )
    )
    """

    def __init__(
        self,
        #env: str,
        batch_size: int = 32, # TODO why is this propogated to the agent?
        replay_size: int = 1_000_000,
        warm_start_steps: int = 200_000,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_last_frame: int = 1_000_000,
        sync_rate: int = 5_000,
        lr: float = 1e-3,
        episode_length: int = 100_000,
        stack_length: int = 2,
        net_kwargs: Dict = None,
        battle_format: str = 'gen7randombattle',
        n_battles: int = 200,
        eval_interval: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.replay_size = replay_size
        self.warm_start_steps = warm_start_steps
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_last_frame = eps_last_frame
        self.sync_rate = sync_rate
        self.lr = lr
        self.episode_length = episode_length
        self.batch_size = batch_size
        self.stack_size = stack_length
        self.battle_format = battle_format
        self.n_battles = n_battles
        self.eval_interval = eval_interval

        '''
        self.logger.log_hyperparams(
            dict(
                replay_size=self.replay_size,
                warm_start_steps=self.warm_start_steps,
                gamma=self.gamma,
                eps_start=self.eps_start,
                eps_end=self.eps_end,
                eps_last_frame=self.eps_last_frame,
                sync_rate=self.sync_rate,
                lr=self.lr,
                episode_length=self.episode_length,
                batch_size=self.batch_size,
                stack_size=self.stack_size,
                battle_format=self.battle_format,
                n_battles=self.n_battles,
            )
        )
        '''
        #self.loop = asyncio.new_event_loop()
        #asyncio.set_event_loop(self.loop)

        #self.env = gym.make(env)
        #self.env = env
        #obs_size = self.env.observation_space.shape[0]

        # TODO can do priority buffer since it's already here
        self.buffer = ReplayBuffer(self.replay_size)
        self.total_reward = 0
        self.episode_reward = 0
        self.eval_opponents = None
        self.eval_player = None
        # Delay setting up
        self.net = None
        self.target_net = None
        self.agent = None
        self.val_opponent = None
        self.net_kwargs = net_kwargs or {}
        self.best_win_rate = -1
        self.best_net = None

        self.net = DQN(**self.net_kwargs)
        #from torchsummary import summary
        #summary(self.net.to('cuda'), (obs_size, self.stack_length)) 
        self.target_net = DQN(**self.net_kwargs)

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        eval_net = deepcopy(self.net)
        eval_net.cpu()
        eval_net.eval()

        self.eval_player = DQNPlayer(
                battle_format=self.battle_format, 
                net=eval_net,
                stack_size=self.stack_size,
                # TODO test removing
                max_concurrent_battles=self.n_battles,
        )
        self.eval_opponents = [
            (name, const(
                battle_format=self.battle_format, 
                max_concurrent_battles=self.n_battles
            )) 
            for name, const in [("max_damage", MaxDamagePlayer), ("random", RandomPlayer)]
        ]
        #self.opponent.policy = utils.build_random_policy(n_actions)
        # TODO copy
 
    def build(self, env, opponent, val_env=None, val_opponent=None):
        self.opponent = opponent
        self.opponent.update_policy(self.target_net, 1.0)
        self.agent = Agent(env, self.buffer, stack_length=self.stack_size)
        self.populate(self.warm_start_steps)

    def update_opponent(self, opponent):
        self.opponent = opponent

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.
        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes in a state `x` through the network and gets the `q_values` of each action as an output.
        Args:
            x: environment state
        Returns:
            q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.
        Args:
            batch: current mini batch of replay data
        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values_idx = self.target_net(next_states).argmax(1)
            next_state_values = self.net(next_states)
            next_state_values = next_state_values.gather(1, next_state_values_idx.unsqueeze(-1)).squeeze(-1)
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def dqn_huber_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Calculates the huber loss using a mini batch from the replay buffer.
        Args:
            batch: current mini batch of replay data
        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values_idx = self.target_net(next_states).argmax(1)
            next_state_values = self.net(next_states)
            next_state_values = next_state_values.gather(1, next_state_values_idx.unsqueeze(-1)).squeeze(-1)
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + rewards
        return F.smooth_l1_loss(state_action_values, expected_state_action_values)

        #states, actions, rewards, dones, next_states = batch

        #state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        #next_state_action_values = self.net(next_states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        #best_action_idx = next_state_action_values.argmax(1)

        #with torch.no_grad():
            #next_state_values = self.target_net(next_states).max(1)[0]
            #next_state_values_action = self.target_net(next_states).max(1)[0]
            #next_state_values = self.target_net(next_states)[best_action_idx]
            #next_state_values[dones] = 0.0
            #next_state_values = next_state_values.detach()

        #expected_state_action_values = next_state_values * self.gamma + rewards

        #return F.smooth_l1_loss(state_action_values, expected_state_action_values)
    
    # Using custom or multiple metrics (default_hp_metric=False)
    def on_fit_end(self):
        self.loop.close()

    def on_train_start(self):
        self.logger.log_graph(self.net)
        #self.logger.log_hyperparams(self.hparams, self.hparams)
        self.logger.log_hyperparams(self.hparams, {
            #"hp/replay_size": self.replay_size,
            "hp/batch_size": self.batch_size,
            "hp/stack_size": self.stack_size,
            "hp/gamma": self.gamma,
            "hp/lr": self.lr,
            "hp/episode_length": self.episode_length,
            "hp/warm_start_steps": self.warm_start_steps,
            "hp/sync_rate": self.sync_rate,
            "hp/eps_start": self.eps_start,
            "hp/eps_end": self.eps_end,
            "hp/eps_last_frame": self.eps_last_frame,
        })
        #self.logger.log_hyperparams(, {"hp/metric_1": 0, "hp/metric_2": 0})

    def evaluate_battler(self, log=True):
        self.eval_player.net = deepcopy(self.net)
        self.eval_player.net.cpu()
        self.eval_player.net.eval()
        for name, opp in self.eval_opponents:
            # TODO pull out to function, don't wont to recreate agent every time 
            self.eval_player.reset_battles()
            self.loop.run_until_complete(self.eval_player.battle_against(opp, n_battles=self.n_battles))

            results = {
                f'{name}/wins': self.eval_player.n_won_battles,
                f'{name}/finished': self.eval_player.n_finished_battles,
                f'{name}/ties': self.eval_player.n_tied_battles,
                f'{name}/losses': self.eval_player.n_lost_battles,
                f'{name}/win_rate': self.eval_player.win_rate,
            }
            # TODO 
            if name == 'max_damage' and self.eval_player.win_rate > self.best_win_rate:
                self.best_win_rate = self.eval_player.win_rate
                self.best_net = deepcopy(self.net)

            if log:
                for key, value in results.items():
                    self.log(key, value)
            #print(results)

    def on_train_end(self):
        """Evaluates the DQN agent on the environment.
        Args:
            net: network to evaluate
            n_episodes: number of episodes to evaluate for
            render: whether to render the environment
        Returns:
            mean reward over the episodes
        """
        pass
        self.evaluate_battler(False)

    # TODO battle format
    def training_epoch_end(self, outputs) -> None:
        #print(outputs)
        #print(len(outputs))
        #print(type(outputs))
        #self.logger.log_graph(self.net)
        #for item in outputs:
            #for key, value in item['log'].items():
                #self.log(key, value)
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_loss', avg_loss)
        self.log('epsilon', self.epsilon)
        self.log('target_updates', self.global_step // self.sync_rate)
        #loop = asyncio.new_event_loop()
        #asyncio.set_event_loop(loop)
        #self.log('agent', self.agent.log())
        for key, value in self.agent.log().items():
            self.log(key, value)

        #if self.global_step % (40*self.sync_rate) == 0:
        if self.current_epoch % self.eval_interval == 0:
            self.evaluate_battler()

    @property
    def epsilon(self) -> float:
        return max(self.eps_end, self.eps_start - ((self.global_step + 1) / self.eps_last_frame))

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch received.
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number
        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        epsilon = self.epsilon

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward

        # calculates training loss
        #loss = self.dqn_mse_loss(batch)
        loss = self.dqn_huber_loss(batch)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())
            self.opponent.update_policy(self.target_net, epsilon)

        log = {
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.tensor(reward).to(device),
            "steps": torch.tensor(self.global_step).to(device),
            "epsilon": torch.tensor(epsilon).to(device),
        }

        # TODO log this more often to observe surprise
        #self.log('loss', loss)

        #self.log('total_reward', self.total_reward)
        #self.log('steps', self.global_step)
        #self.log('epsilon', epsilon)

        return OrderedDict({"loss": loss, "log": log, "progress_bar": log})

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer, self.episode_length)
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.batch_size, 
            #sampler=None,
            persistent_workers=True,
            num_workers=1, # is this cuasing it to only run less iterations?
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = parent_parser.add_argument_group("DQNLightning")
        parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
        parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
        parser.add_argument("--env", type=str, default="CartPole-v1", help="gym environment tag")
        parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
        parser.add_argument("--sync_rate", type=int, default=10, help="how many frames do we update the target network")
        parser.add_argument("--replay_size", type=int, default=1000, help="capacity of the replay buffer")
        parser.add_argument(
            "--warm_start_steps",
            type=int,
            default=1000,
            help="how many samples do we use to fill our buffer at the start of training",
        )
        parser.add_argument("--eps_last_frame", type=int, default=1000, help="what frame should epsilon stop decaying")
        parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
        parser.add_argument("--eps_end", type=float, default=0.01, help="final value of epsilon")
        parser.add_argument("--episode_length", type=int, default=200, help="max length of an episode")
        return parent_parser
