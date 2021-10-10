import asyncio
import gym
from multiprocessing import Array, Process
from threading import Thread, stack_size
from typing import List
from numpy.core.numeric import indices
from poke_env.player.player import Player

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, EventCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import battler # registers gym env
import battler.players
from poke_env.player.baselines import MaxBasePowerPlayer, DummyPlayer
#from poke_env.player.env_player import LOOP
import asyncio
import numpy as np
from icecream import ic

from stable_baselines3.common.callbacks import BaseCallback
from copy import deepcopy
from battler.utils.embed import embed_battle
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from battler.models.ppo_models import PokeMLP
from stable_baselines3.common.env_checker import check_env
from collections import deque

def pls(model):
    def policy(observation):
        return model.predict(observation)
    return policy

class Opp(DummyPlayer):
    def set_policy(self, policy):
            self.policy = lambda x: policy.predict(x, None, None, None)
class SelfPlayCallback(BaseCallback):
    def __init__(
        self, 
        envs,
        policy_creator, 
        starting_policy=None,
        *, 
        opponent_shuffle_freq=1000,
        reward_threshold=0.55,
        max_policies=100,
        **kwargs
    ):

        # Save reference to opponent
        ic(envs)
        self.envs = envs
        self.policy_creator = policy_creator
        self.policies = deque([starting_policy], maxlen=max_policies)
        self.opponent_shuffle_freq = opponent_shuffle_freq
        self.policy_switches = 0
        self.policy_pushes = 0


        super(SelfPlayCallback, self).__init__(
            self,
            #self.add_policy, 
            **kwargs)

    @staticmethod
    def create(model):
        pass

    def add_policy(self):
        ic()
        # TODO since it's picklying and sending to subprocesses, do I need to copy?
        #policy = deepcopy(self.model.policy)
        # TODO is eval or other stuff needed?
        #def predict(x):
            #return policy.predict(x, None, None, None)
            #return policy.predict(x, None, None, None)
        self.policies.append(deepcopy(self.model.policy))
        self.policy_pushes += 1
    
    
    def on_rollout_end(self) -> None:
        ic()
        ic(self.logger.name_to_value)
        self.add_policy()
        self.logger.record("self_play/n_policies", len(self.policies))
        self.logger.record("self_play/policy_switches", self.policy_switches)
        #if self.model.logger.name_to_value[]
        #ic(self.globals)
        #$ic(self.locals)
        #if 'train/'
        return super().on_rollout_end()

    # TODO should dummy player hold policies so I don't have to reach for them?
    # TODO should wrap dummy player?
    def switch_opponents_policies(self):
        polices = np.random.choice(self.policies, size=self.envs.num_envs, replace=True) # TODO Weight based on eval
        self.policy_switches += 1
        # maybe no race condition as opps are not currenlty moving
        for idx, policy in enumerate(polices):
            self.envs.env_method('set_opponent_policy', indices=idx, policy=policy)

    def _on_step(self) -> bool:
        if self.opponent_shuffle_freq > 0 and self.opponent_shuffle_freq % self.n_calls == 0:
            self.switch_opponents_policies()

        return super()._on_step()


# TODO pull out action to move to wrapper between move maker to reweight based on legal moves
def policy_creator(model):
    def policy(observation):
        return model.predict(observation)
    return policy

# TODO move poke-envs to dict spaces

from stable_baselines3.common.vec_env import VecFrameStack
N_ENVS = 4
def train():
    #env = make_vec_env(
        #'Pokemon-v8', 
        #n_envs=2,
        #env_kwargs={
            #"battler_embedder": lambda _: np.array([0., 0.1, 2., 0.1])
        #}
    # NOTE can't make n_envs bigger than batch_size?
    check_env(gym.make('Pokemon-v8'))
    env = make_vec_env(
        'Pokemon-v8', 
        n_envs=N_ENVS, 
        seed=0, 
        vec_env_cls=SubprocVecEnv, 
        env_kwargs=dict(
            opponent_cls=Opp, 
            battle_embedder=embed_battle, 
            stack_size=1))
    ic(env.observation_space)
    #env = VecFrameStack(env, n_stack=32, channels_order='first')
    ic(env.observation_space)

    #eval_env = make_vec_env('Pokemon-v8', n_envs=1)
    eval_env = make_vec_env(
        'Pokemon-v8', 
        n_envs=max(1, N_ENVS//4), 
        seed=0, 
        #n_eval_episodes=1000,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=dict(
            opponent_cls=MaxBasePowerPlayer,
            opponent_kwargs=dict(
                battle_format='gen8randombattle')
        ))
    # TODO pull out embedings since procs seem to rebuilding them
    #eval_env = VecFrameStack(eval_env, n_stack=4, channels_order='first')
    opp_update_cb = SelfPlayCallback(
        starting_policy=None,
        policy_creator=policy_creator,
        # NOTE can't retrieve opponent due to event loop
        #opponents_policy_setter=env.get_attr('get_opponent_policy_setter'),
        envs=env,
        #opponents=[],
        opponent_shuffle_freq=10_000, # 10000 ~= 2000 games (<60 moves)  
    )

    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./logs/',
        n_eval_episodes=500,
        log_path='./logs/', 
        eval_freq=max(250_000 // N_ENVS, 1),
        deterministic=False,  # need stocastic actions due to nature of pokemon battles
        render=False,
        callback_on_new_best=None)


    policy_kwargs = dict(
        features_extractor_class=PokeMLP,
        features_extractor_kwargs=dict(hidden_dim=256, n_actions=128)
    )
    # TODO create callback fro copying over new policy
    # TODO should only a single opponent exist and accept all challanges
    # TODO lstm policy on top of features extractor
    model = PPO(
        'MlpPolicy', 
        env, 
        policy_kwargs=policy_kwargs,
        batch_size=max(N_ENVS, 512),
        n_steps=8192//N_ENVS,
        tensorboard_log='tensorboard_log/ppo',
        verbose=1)

    model.learn(
        total_timesteps=8000000,
        #callback=eval_callback,
        callback=[eval_callback, opp_update_cb],
    )
    # TODO don't use dummy vec. use process vec!!!!

if __name__ == '__main__':
    train()