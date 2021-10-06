import asyncio
import gym
from multiprocessing import Array, Process
from threading import Thread

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import battler # registers gym env
import battler.players
from poke_env.player.baselines import MaxBasePowerPlayer
#from poke_env.player.env_player import LOOP
import asyncio
import numpy as np
from icecream import ic

def train():
    #env = make_vec_env(
        #'Pokemon-v8', 
        #n_envs=2,
        #env_kwargs={
            #"battler_embedder": lambda _: np.array([0., 0.1, 2., 0.1])
        #}
    env = make_vec_env('Pokemon-v8', n_envs=32, seed=0, vec_env_cls=SubprocVecEnv)
    #eval_env = make_vec_env('Pokemon-v8', n_envs=1)
    #eval_env = gym.make('Pokemon-v8') #, opponent=MaxBasePowerPlayer('gen8randombattle'))
    #eval_env.set_opponent(MaxBasePowerPlayer(battle_format='gen8randombattle'))
    # Use deterministic actions for evaluation
    '''
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                log_path='./logs/', eval_freq=10_000,
                                deterministic=True, render=False,
                                callback_on_new_best=None)
    '''

    # TODO create callback fro copying over new policy
    # TODO should only a single opponent exist and accept all challanges
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(
        total_timesteps=80000,
        #callback=eval_callback,
    )
    # TODO don't use dummy vec. use process vec!!!!

    ic('done learning')
    obs = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

if __name__ == '__main__':
    train()