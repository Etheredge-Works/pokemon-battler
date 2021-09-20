from typing import Callable, List, Tuple
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from poke_env.environment.battle import AbstractBattle
from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer

from battler.utils import embed


def build_static_policy(net: nn.Module, prob: float) -> Callable:
    def policy(battle: AbstractBattle) -> int:
        if np.random.random() < prob:
            return np.random.randint(net.action_size)
        else:
            encoded_state = torch.from_numpy(embed.embed_battle(battle))
            return net(encoded_state).argmax().item()
    return policy


def build_stocastic_policy(net: nn.Module, prob: float = 1.0) -> Callable:
    def policy(battle: AbstractBattle) -> int:
        if np.random.random() < prob:
            return np.random.randint(net.action_size)
            # TODO self.choos_random_move(battle)?
        else:
            encoded_state = embed.embed_battle(battle)
            y = net(torch.from_numpy(encoded_state))
            probabilities = F.softmax(y, dim=0)
            return torch.multinomial(probabilities, 1).item()
    return policy

def build_eval_stocastic_policy(net: nn.Module) -> Callable:
    def policy(battle: AbstractBattle) -> int:
        encoded_state = embed.embed_battle(battle)
        y = net(torch.from_numpy(encoded_state))
        probabilities = F.softmax(y, dim=0)
        return torch.multinomial(probabilities, 1).item()
    return policy


class RandomOpponentPlayer(RandomPlayer):
    def update_policy(self, net: nn.Module, _: float = 1.0):
        pass

 
class RLOpponentPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = None

    def choose_move(self, battle: AbstractBattle):
        action = self.policy(battle)
        return self.create_order(action)
    
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

    def update_policy(self, net: nn.Module, _: float = 1.0):
        pass

