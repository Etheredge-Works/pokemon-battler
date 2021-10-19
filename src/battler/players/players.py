from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from poke_env.environment.battle import Battle, AbstractBattle
from poke_env.player.env_player import Gen7EnvSinglePlayer, Gen8EnvSinglePlayer
from typing import Callable
from ..utils.embed import embed_battle, BATTLE_SHAPE
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
from poke_env.player.battle_order import BattleOrder
import gym


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


class RLPlayer7(Gen7EnvSinglePlayer):
    def __init__(
        self, 
        reward_kwargs: Dict = None,
        stack_size: int = 1,
        battle_format: str = 'gen7randombattle',
        **kwargs,
    ):
        super().__init__(battle_format=battle_format, **kwargs)
        # take out z moves, mega moves, and dyna moves

        # TODO reduce action space
        # TODO pokemon sorting based on switches
        self._action_space = spaces.Discrete(len(self._ACTION_SPACE))

        self.reward_kwargs = reward_kwargs or {}
        self.state = None
        self.stack_size = stack_size

        self._observation_space = spaces.Box(
            float("-inf"), float("inf"), shape=(stack_size, BATTLE_SHAPE,))
        
    def _action_to_move(self, action: int, battle: Battle) -> BattleOrder:
        #if action >= 4:
            #action += self.removed_actions_delta

        return super()._action_to_move(action, battle)

    @property
    def observation_space(self) -> np.array:
        return self._observation_space

    @property
    def action_space(self) -> List:
        """Returns the action space of the player. Must be implemented by subclasses."""
        return self._action_space

    def reset(self):
        self.state = None
        return super().reset()

    def compute_reward(self, battle: Battle) -> float:
        return self.reward_computing_helper(
            battle,
            **self.reward_kwargs,
        )

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        battle_embedding = embed_battle(battle)
        if self.state is None:
            self.state = deque([battle_embedding for _ in range(self.stack_size)], maxlen=self.stack_size)
        else:
            self.state.append(battle_embedding)
        return np.stack(self.state)

class WrappedPlayer(gym.Wrapper):
    def __init__(self, env, **kwargs):
        super().__init__(env)
        self.kwargs = kwargs

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def __getattr__(self, name):
        return getattr(self.env, name)

def battle_stacker_builder(stack_size):
    stacked_state = None
    def embed(battle: AbstractBattle) -> np.ndarray:
        nonlocal stacked_state
        eb = embed_battle(battle)
        if stacked_state is None:
            stacked_state = deque([eb for _ in range(stack_size-1)], maxlen=stack_size)
        stacked_state.append(eb)
        result = np.stack(stacked_state)
        if battle.finished:
            stacked_state = None
        return result
    return embed
class RLPlayer8(Gen8EnvSinglePlayer):
    def __init__(
        self, 
        reward_kwargs: Dict = None,
        stack_size: int = 1,
        battle_format: str = 'gen8randombattle',
        **kwargs,
    ):
        super().__init__(battle_format=battle_format, **kwargs)
        # take out z moves, mega moves, and dyna moves
        #self.removed_actions_delta = 4 * 3
        #last_action = len(self._ACTION_SPACE) - self.removed_actions_delta
        #self._ACTION_SPACE = range(0, last_action)

        # TODO reduce action space
        # TODO pokemon sorting based on switches
        self._action_space = spaces.Discrete(len(self._ACTION_SPACE))

        self.reward_kwargs = reward_kwargs or {}
        self.state = None
        self.stack_size = stack_size

        self._observation_space = spaces.Box(
            float("-inf"), float("inf"), shape=(stack_size, BATTLE_SHAPE,))
            #float("-inf"), float("inf"), shape=(stack_size, BATTLE_SHAPE,))
        
    def _action_to_move(self, action: int, battle: Battle) -> BattleOrder:
        #if action >= 4:
            #action += self.removed_actions_delta

        return super()._action_to_move(action, battle)

    def _battle_finished_callback(self, _):
        super()._battle_finished_callback(_)
        self.state = None

    def get_opponent_policy_setter(self):
        return self._opponent.set_policy

    def set_opponent_policy(self, policy):
        self._opponent.set_policy(policy)

    @property
    def observation_space(self) -> np.array:
        return self._observation_space

    @property
    def action_space(self) -> List:
        """Returns the action space of the player. Must be implemented by subclasses."""
        return self._action_space

    def reset(self):
        self.state = None
        r = super().reset()
        #ic(r.shape)
        #self.state = deque([battle_embedding for _ in range(self.stack_size)], maxlen=self.stack_size)
        return r

    def compute_reward(self, battle: Battle) -> float:
        return self.reward_computing_helper(
            battle,
            **self.reward_kwargs,
        )

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        # TODO stacking should be done with a wrapper
        battle_embedding = embed_battle(battle)
        #er = np.expand_dims(er, axis=0)
        #return er
        #return np.stack([battle_embedding]).swapaxes(0, 1)
        if self.state is None:
            self.state = deque([battle_embedding for _ in range(self.stack_size-1)], maxlen=self.stack_size)
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
        
