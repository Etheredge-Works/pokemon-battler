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
from icecream import ic


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

import gym
gym.register(
    id='Pokemon-v8',
    entry_point='battler.players.players:RLPlayer',
    max_episode_steps=1000,
    nondeterministic=True,
)
class RLPlayer(Gen8EnvSinglePlayer):
    def __init__(
        self, 
        reward_kwargs: Dict = None,
        obs_space: int = 1,
        stack_size: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.obs_space = obs_space
        self._action_space = spaces.Discrete(len(self._ACTION_SPACE))

        self.reward_kwargs = reward_kwargs or {}
        self.state = None
        self.stack_size = stack_size
        self._observation_space = spaces.Box(
            float("-inf"), float("inf"), shape=(stack_size, self.obs_space,))
        
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
            **self.reward_kwargs,
            #victory_value=30,
            #fainted_value=-2,
            #hp_value=2,
            #status_value=0.5
            
        )
        starting_value = 0
        number_of_pokemons = 6

        prorated_value = 6 
        victory_value = 1
        fainted_value = prorated_value / number_of_pokemons
        status_value = fainted_value / 5
        hp_value = fainted_value - status_value

        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = starting_value
        current_value = 0

        #for mon in battle.team.values():
            #current_value += mon.current_hp_fraction * hp_value
            #if mon.fainted:
                #current_value -= fainted_value
            #elif mon.status is not None:
                #current_value -= status_value

        #current_value += (number_of_pokemons - len(battle.team)) * hp_value

        for mon in battle.opponent_team.values():
            #current_value += 1/ (mon.current_hp_fraction * hp_value)
            #damange_done = mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value += fainted_value
            else:
                if mon.status is not None:
                    current_value += status_value
                # 1.0 -> 0.0 * hp_value
                # 0.0 -> 1 * hp_value
                # 0.2 -> 0.8 * hp_value
                current_value += (1 - mon.current_hp_fraction) * hp_value


        # NOTE need some victory value or agent could delay the game more? 
        #      shouldn't since it wants the W sooner foro the hp reward
        #if battle.won:
            #current_value += victory_value
        #elif battle.lost:
            #current_value -= victory_value

        to_return = current_value - self._reward_buffer[battle]
        self._reward_buffer[battle] = current_value

        return to_return

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        battle_embedding = embed_battle(battle)
        if self.state is None:
            self.state = deque([battle_embedding for _ in range(self.stack_size)], maxlen=self.stack_size)
        else:
            self.state.append(battle_embedding)
        #return self.state
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

#class VectorPlayer(gym.ObservationWrapper):
    #def __init__(self, env: gym.Env, n_obs: int = 1, **kwargs):

#class VectorizedPlayer(gym.vector.VectorEnv):
class VectorizedPlayer(gym.Env):
    def __init__(self, env_name: str, n_obs: int = 1, **kwargs):
        super().__init__()

        self.envs = [gym.make(env_name) for _ in range(n_obs)]
        self.n_obs = n_obs
        #self.observation_space = spaces.Box(
            #float("-inf"), float("inf"), shape=(n_obs, self..shape[0]))
        
