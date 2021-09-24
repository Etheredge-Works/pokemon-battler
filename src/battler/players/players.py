from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from poke_env.environment.battle import Battle, AbstractBattle
from poke_env.player.env_player import Gen7EnvSinglePlayer, Gen8EnvSinglePlayer
from typing import Callable
from ..utils.embed import embed_battle
import numpy as np
from typing import List, Dict
from gym.spaces import Discrete
from gym import spaces
from battler import utils

import torch
from torch import nn
from torch.nn import functional as F
from collections import deque


def build_static_policy(net: nn.Module, prob: float) -> Callable:
    def policy(battle: AbstractBattle) -> int:
        if np.random.random() < prob:
            return np.random.randint(net.action_size)
        else:
            encoded_state = embed_battle(battle)
            return net(encoded_state).argmax().item()
    return policy


def build_stocastic_policy(net: nn.Module, prob: float = 1.0) -> Callable:
    def policy(battle: AbstractBattle) -> int:
        if np.random.random() < prob:
            return np.random.randint(net.action_size)
        else:
            encoded_state = embed_battle(battle)
            y = net(encoded_state).item()
            probabilities = F.softmax(y)
            return torch.multinomial(probabilities, 1).item()
    return policy


'''
class RandomOpponentPlayer(Player):
    def choose_move(self, battle: AbstractBattle):
        return self.create_order(self.policy(battle))
    
    def update_policy(self, net: nn.Module, prob: float = 1.0):
        self.policy = build_static_policy(net, 1.0)

'''

 
class RLOpponentPlayer(Player):
    def choose_move(self, battle: AbstractBattle):
        if self.policy is None:
            return self.choose_random_move(battle)
        return self.create_order(self.policy(battle))
    
    def update_policy(self, policy):
        #self.policy = policy
        pass
        

class MaxDamagePlayer(RandomPlayer):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


class RLPlayer(Gen8EnvSinglePlayer):
    def __init__(
        self, 
        reward_kwargs: Dict = {},
        obs_space: int = 1,
        action_space: int = 17, # gen 7
        stack_size: int = 1,
        # TODO kwargs these things
        #fainted_value: float = 0.0,
        #hp_value: float = 0.0,
        #number_of_pokemons: int = 6,
        #starting_value: float = 0.0,
        #status_value: float = 0.0,
        #victory_value: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.obs_space = obs_space
        self._action_space = spaces.Discrete(len(self._ACTION_SPACE))
        #self._action_space = spaces.Discrete(action_space)
        #self.fainted_value = fainted_value
        #self.hp_value = hp_value
        #self.number_of_pokemons = number_of_pokemons
        #self.starting_value = starting_value
        #self.status_value = status_value
        #self.victory_value = victory_value
        self.reward_kwargs = reward_kwargs
        self.state = None
        self.stack_size = stack_size
        self._observation_space = spaces.Box(
            float("-inf"), float("inf"), shape=(self.obs_space,))
            #float("-inf"), float("inf"), shape=(self.stack_size * self.obs_space,))
            #kfloat("-inf"), float("inf"), shape=(self.stack_size, self.obs_space,))

    @property
    def observation_space(self) -> np.array:
        return self._observation_space

    @property
    def action_space(self) -> List:
        """Returns the action space of the player. Must be implemented by subclasses."""
        return self._action_space
        #return spaces.Discrete(21)
        #return spaces.Discrete(self._ACTION_SPACE)

    def reset(self):
        self.state = None
        return super().reset()

    def compute_reward(self, battle: Battle) -> float:
        return self.reward_computing_helper(
            battle,
            #**self.reward_kwargs
            #fainted_value=self.fainted_value,
            #hp_value=self.hp_value,
            #number_of_pokemons=self.number_of_pokemons,
            #starting_value=self.starting_value,
            #status_value=self.status_value,
            #victory_value=self.victory_value
        )

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        battle_embedding = embed_battle(battle)
        if self.state is None:
            self.state = deque([battle_embedding for _ in range(self.stack_size)], maxlen=self.stack_size)
        else:
            self.state.append(battle_embedding)
        return self.state
        #return battle_embedding
        return np.stack(self.state)

    '''
    def play_against(
        self, 
        env_algorithm: Callable, 
        opponent: Player, 
        env_algorithm_kwargs=None
    ) -> AbstractBattle:

        return super().play_against(opponent, battle)
    '''