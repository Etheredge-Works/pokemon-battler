import asyncio
import gym
from multiprocessing import Array, Process
from threading import Thread, stack_size
from typing import List
from numpy.core.numeric import indices
from poke_env.environment.battle import Gen8Battle
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
from battler.models.ppo_models import PokeMLP, Embeddings, MovesNet, PokesNet
from stable_baselines3.common.env_checker import check_env
from collections import deque
from stable_baselines3.common.vec_env import VecFrameStack
from torchsummary import summary
from poke_env.player.env_player import Gen8EnvSinglePlayer


class SelfPlayCallback(BaseCallback):
    # TODO does having enough policies negate the need to only push better policies?
    def __init__(
        self, 
        envs,
        policy_creator, 
        starting_policy=None,
        *, 
        opponent_shuffle_freq=1000,
        win_rate_threshold=0.55,
        max_policies=500,
        **kwargs
    ):

        # Save reference to opponent
        ic(envs)
        self.envs = envs
        #self.policy_creator = policy_creator
        self.policies = deque([starting_policy], maxlen=max_policies)
        self.opponent_shuffle_freq = opponent_shuffle_freq
        self.policy_switches = 0
        self.policy_pushes = 0

        super(SelfPlayCallback, self).__init__(
            self,
            #self.add_policy, 
            **kwargs)

        self.current_player = DummyPlayer(
            battle_embedder=embed_battle,
            action_to_move=Gen8EnvSinglePlayer._action_to_move,
        )
        self.current_player.set_policy(self.model.policy)
        self.old_player = DummyPlayer(
            battle_embedder=embed_battle,
            action_to_move=Gen8EnvSinglePlayer._action_to_move,
        )
        self.win_rate_threshold = win_rate_threshold

    @staticmethod
    def create(model):
        pass

    def battle_old_self(self):
        self.current_player.reset_battles()
        self.current_player.battle_against(self.old_player, 1000)
        return self.current_player.win_rate()

    def add_policy(self):
        ic()
        # TODO since it's picklying and sending to subprocesses, do I need to copy?
        #policy = deepcopy(self.model.policy)
        # TODO is eval or other stuff needed?
        #def predict(x):
            #return policy.predict(x, None, None, None)
            #return policy.predict(x, None, None, None)
        policy_copy = deepcopy(self.model.policy)
        self.policies.append(policy_copy) # TODO get policies on cpu...
        self.old_player.set_policy(policy_copy)

        # Have to switch after to avoid segfault from old policy
        self.switch_opponents_policies()

        #self.policies.append(deepcopy(self.model.policy)) # TODO get policies on cpu...
        #self.policies.append(deepcopy(self.model.policy).cpu()) # TODO get policies on cpu...
        self.policy_pushes += 1
    
    
    def on_rollout_end(self) -> None:
        ic()
        ic(self.logger.name_to_value)
        win_rate = self.battle_old_self()
        self.logger.record('self_play/win_rate_vs_previous', win_rate)
        if win_rate > self.win_rate_threshold:
            self.add_policy()
        self.logger.record("self_play/n_policies", len(self.policies))
        self.logger.record("self_play/policy_switches", self.policy_switches)
        self.logger.record("self_play/policy_pushes", self.policy_pushes)
        #if self.model.logger.name_to_value[]
        #ic(self.globals)
        #$ic(self.locals)
        #if 'train/'
        return super().on_rollout_end()

    # TODO should dummy player hold policies so I don't have to reach for them?
    # TODO should wrap dummy player?
    def switch_opponents_policies(self):
        polices = np.random.choice(
            self.policies, 
            size=self.envs.num_envs, 
            replace=True) # TODO Weight based on eval
        self.policy_switches += 1
        # maybe no race condition as opps are not currenlty moving
        for idx, policy in enumerate(polices):
            if policy is not None:
                #policy.cpu()
                pass
            self.envs.env_method('set_opponent_policy', indices=idx, policy=policy)

    def _on_step(self) -> bool:
        if self.opponent_shuffle_freq > 0 and self.n_calls % self.opponent_shuffle_freq:
            self.switch_opponents_policies()

        return super()._on_step()


# TODO could do boosting style with policies based on win rate?
# RESEARCH TODO switching policies within games allow for more diverse strategies?
# RESEARCH TODO is this similar to combining polcies? like genetic algorithm?
# TODO can write get winrate for envs?


# TODO pull out action to move to wrapper between move maker to reweight based on legal moves
def policy_creator(model):
    def policy(observation):
        return model.predict(observation)
    return policy

# TODO move poke-envs to dict spaces

N_ENVS = 8
# TODO why does going from 8->32 make python gpu processes expload?
stack_size = 16
format = "gen8randombattle"
#env_cls = DummyVecEnv
env_cls = SubprocVecEnv
def train():
    #env = make_vec_env(
        #'Pokemon-v8', 
        #n_envs=2,
        #env_kwargs={
            #"battler_embedder": lambda _: np.array([0., 0.1, 2., 0.1])
        #}
    # NOTE can't make n_envs bigger than batch_size?
    #check_env(gym.make('Pokemon-v8', stack_size=10))
    env = make_vec_env(
        'Pokemon-v8', 
        n_envs=N_ENVS, 
        seed=0, 
        vec_env_cls=env_cls, 
        env_kwargs=dict(
            opponent_cls=DummyPlayer, 
            opponent_kwargs=dict(
                battle_format=format,
                battle_embedder=embed_battle, 
                action_to_move=Gen8EnvSinglePlayer._action_to_move,
                #battle_embedder=lambda x: torch.tensor(embed_battle(x)), 
                stack_size=stack_size,
            ),
            battle_embedder=embed_battle, 
            stack_size=stack_size,
        ),
    )
    ic(env.observation_space)
    #env = VecFrameStack(env, n_stack=32, channels_order='first')
    ic(env.observation_space)

    #eval_env = make_vec_env('Pokemon-v8', n_envs=1)
    eval_env = make_vec_env(
        'Pokemon-v8', 
        n_envs=max(1, N_ENVS), 
        seed=0, 
        vec_env_cls=env_cls,
        env_kwargs=dict(
            opponent_cls=MaxBasePowerPlayer,
            opponent_kwargs=dict(
                battle_format='gen8randombattle'),
            battle_embedder=embed_battle, 
            stack_size=stack_size,
        )
    )
    # TODO pull out embedings since procs seem to rebuilding them
    #eval_env = VecFrameStack(eval_env, n_stack=4, channels_order='first')
    # TODO make eval callback with previous policy? or with shuffle?
    opp_update_cb = SelfPlayCallback(
        starting_policy=None,
        policy_creator=policy_creator,
        # NOTE can't retrieve opponent due to event loop
        #opponents_policy_setter=env.get_attr('get_opponent_policy_setter'),
        envs=env,
        #opponents=[],
        opponent_shuffle_freq=400, # 10000 ~= 2000 games (<60 moves)  
    )
    # TODO smart action chooser in env. take action probs, and reweight based on legal moves
    # TODO for eval, could just create a dummy with current policy and have battle a ton
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./logs/',
        n_eval_episodes=500,
        log_path='./logs/', 
        eval_freq=max(80_000 // N_ENVS, 1),
        deterministic=False,  # need stocastic actions due to nature of pokemon battles
        render=False,
        callback_on_new_best=None)

    # TODO could send weights to dummy player instead of models?
    e = Embeddings()
    ic(e)
    pn = PokesNet(
        name_embedding=e.name_embedding,
        status_embedding=e.status_embedding,
        type_embedding=e.type_embedding,
        ability_embedding=e.ability_embedding,
        item_embedding=e.item_embedding,
    )
    mn = MovesNet(
        type_embedding=e.type_embedding,
    )
    policy_kwargs = dict(
        features_extractor_class=PokeMLP,
        features_extractor_kwargs=dict(
            weather_embedding=e.weather_embedding,
            moves_net=mn,
            pokes_net=pn,
            hidden_dim=128, 
            n_actions=128)
    )

    # TODO for torch, deep is way slower than wide...

    # TODO create callback fro copying over new policy
    # TODO should only a single opponent exist and accept all challanges
    # TODO lstm policy on top of features extractor
    # TODO it's probably not bad to just test against previous self, since it will be learning from old selves
    model = PPO(
        'MlpPolicy', 
        env, 
        policy_kwargs=policy_kwargs,
        batch_size=max(N_ENVS, 64),
        n_epochs=10,
        # If switching 
        #n_steps=16384//N_ENVS,
        n_steps=8192//N_ENVS,
        tensorboard_log='tensorboard_log/ppo',
        verbose=1)
    summary(model.policy, env.observation_space.shape)

    ic(model.policy)

    model.learn(
        total_timesteps=8000000,
        #callback=eval_callback,
        callback=[eval_callback, opp_update_cb],
    )
    # TODO don't use dummy vec. use process vec!!!!

if __name__ == '__main__':
    train()