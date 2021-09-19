from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from poke_env.environment.battle import Battle, AbstractBattle
from poke_env.player.env_player import Gen7EnvSinglePlayer
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


class RandomOpponentPlayer(Player):
    def choose_move(self, battle: AbstractBattle):
        return self.create_order(policy(battle))
    
    def update_policy(self, net: nn.Module, prob: float = 1.0):
        self.policy = build_static_policy(net, 1.0)

 
class RLOpponentPlayer(Player):
    def choose_move(self, battle: AbstractBattle):
        return self.create_order(self.policy(battle))
    
    def update_policy(self, net: nn.Module, prob: float = 1.0):
        self.policy = build_stocastic_policy(net, prob)
        

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


class RLPlayer(Gen7EnvSinglePlayer):
    def __init__(
        self, 
        fainted_value: float = 0.0,
        hp_value: float = 0.0,
        #number_of_pokemons: int = 6,
        #starting_value: float = 0.0,
        status_value: float = 0.0,
        victory_value: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fainted_value = fainted_value
        self.hp_value = hp_value
        #self.number_of_pokemons = number_of_pokemons
        #self.starting_value = starting_value
        self.status_value = status_value
        self.victory_value = victory_value

    @property
    def observation_space(self) -> np.array:
        return spaces.Box(float("-inf"), float("inf"), shape=(108,))

    @property
    def action_space(self) -> List:
        """Returns the action space of the player. Must be implemented by subclasses."""
        return spaces.Discrete(17)
        #return spaces.Discrete(21)
        #return spaces.Discrete(self._ACTION_SPACE)

    def compute_reward(self, battle: Battle) -> float:
        return 1
        return self.reward_computing_helper(
            battle,
            fainted_value=self.fainted_value,
            hp_value=self.hp_value,
            #number_of_pokemons=self.number_of_pokemons,
            #starting_value=self.starting_value,
            status_value=self.status_value,
            victory_value=self.victory_value
        )

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        return utils.embed.embed_battle(battle)
