import gym
from multiprocessing import Process
from threading import Thread

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import battler # registers gym env
from poke_env.player.baselines import MaxBasePowerPlayer
from poke_env.player.env_player import LOOP
import asyncio

def train():
    #env = gym.make('CartPole-v1')
    #env = gym.make('Pokemon-v8')
    #env.reset()

    #model = A2C('MlpPolicy', env, verbose=1)

    #eval_env = make_vec_env('Pokemon-v8', n_envs=1)
    env = make_vec_env(
        'Pokemon-v8', 
        n_envs=1,
        #vec_env_cls=SubprocVecEnv,
    )
    #eval_env = make_vec_env('Pokemon-v8', n_envs=1)
    def t():
        #asyncio.set_event_loop(LOOP)
        env.reset()
        env.reset()
        ic('did 2 resets')
        #eval_env = gym.make('Pokemon-v8') #, opponent=MaxBasePowerPlayer('gen8randombattle'))
        env.reset()
        env.reset()
        ic('did 2 more resets')
        eval_env.reset()
        eval_env.reset()
        ic('did 2 eval resets')
        #eval_env.set_opponent(MaxBasePowerPlayer(battle_format='gen8randombattle'))
        # Use deterministic actions for evaluation
        #eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                    #log_path='./logs/', eval_freq=10_000,
                                    #deterministic=True, render=False,
                                    #callback_on_new_best=None)

        # TODO should only a single opponent exist and accept all challanges
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(
            total_timesteps=80000,
            #callback=eval_callback
        )
        # TODO don't use dummy vec. use process vec!!!!

        obs = env.reset()
        for i in range(1000):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()
    t = Thread(target=t)
    t.start()
    LOOP.run_forever()
    t.join()
    #t()

        