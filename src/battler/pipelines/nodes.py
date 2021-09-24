from collections import deque
from poke_env.player.random_player import RandomPlayer
from pytorch_lightning.accelerators import accelerator
from typing import Dict, Tuple
import time
from pytorch_lightning import Trainer
import torch
import numpy as np
from battler.models.dqn import DQN

from battler.models.lightning import DQNLightning
#from battler.players.opponents import #OpponentPlayer

#from poke_env.player.random_player import RandomPlayer
from battler.players.players import RLPlayer
from battler.players.opponents import MaxDamagePlayer, RandomOpponentPlayer, RLOpponentPlayer
import pytorch_lightning as pl


def dqn_training(env, model: pl.LightningModule, trainer: pl.Trainer, opponent):
    torch.manual_seed(0)
    np.random.seed(0)
    model.build(env, opponent)
    trainer.fit(model)

    # This call will finished eventual unfinshed battles before returning
    env.complete_current_battle()


def train(
    battle_format: str,
    opponent_type: str, 
    lightning_kwargs: Dict,
    epochs: int,
    reward_kwargs: Dict,
    net_kwargs: Dict,
    model_path: str,
) -> Dict:
    #import cProfile, pstats, io
    #pr = cProfile.Profile()
    #pr.enable()

    #loop = asyncio.new_event_loop()
    #asyncio.set_event_loop(loop)

    opp = get_opponent(opponent_type)(battle_format=battle_format)
    opp.stack_size = net_kwargs['stack_size'] # TODO clean up

    # TODO move back in side func?
    # Train the model
    trainer = Trainer(
        gpus=1,
        accelerator='dp',
        max_epochs=epochs,
        #log_every_n_steps=1,
        flush_logs_every_n_steps=4_000,
        logger=pl.loggers.TensorBoardLogger(
            '.',
            #log_graph=True,
            default_hp_metric=False,
            #save_dir='tb_logs',
            name='lightning_logs',
        )
    )

    model = DQNLightning(
        **lightning_kwargs,
        net_kwargs=net_kwargs,
    )

    train_kwargs = dict(
        model=model,
        trainer=trainer,
        opponent=opp,
    )

    env_player = RLPlayer(battle_format=battle_format, **reward_kwargs)

    env_player.play_against(
        env_algorithm=dqn_training,
        opponent=opp,
        env_algorithm_kwargs=train_kwargs
    )

    trainer.save_checkpoint(model_path)
    #pr.disable()
    #s = io.StringIO.StringIO()
    #ps = pstats.Stats(pr, stream=s)
    #ps.dump_stats('/tmp/stats.dmp')

    # Return the model
    return dict(
        best_model_state=model.best_net.state_dict(),
        best_model_kwargs=model.best_net.get_kwargs(),
        #model_state=model.net.state_dict(),
        #model_kwargs=model.net.get_kwargs(),
        #model_path=model_path
    )


from battler.utils import get_opponent
from poke_env.player.env_player import Gen7EnvSinglePlayer
from poke_env.player.utils import cross_evaluate
from battler.deploy.dqn_player import DQNPlayer


def evaluate_net(
    model,
    stack_size,
    battle_format,
    opponent_type,
    n_battles,
):
    player = DQNPlayer(battle_format=battle_format, net=model, stack_size=stack_size, max_concurrent_battles=1)
    opponent = get_opponent(opponent_type)(battle_format=battle_format, max_concurrent_battles=1)

    asyncio.get_event_loop().run_until_complete(
        player.battle_against(opponent, n_battles=n_battles))

    results = dict(
        opponent_type=opponent_type,
        wins=player.n_won_battles,
        finished=player.n_finished_battles,
        ties=player.n_tied_battles,
        losses=player.n_lost_battles,
        win_rate=player.win_rate,
    )
    return results

import asyncio

def evaluate(
    model_state,
    model_kwargs: Dict,
    obs_space: int,
    battle_format: str,
    opponent_type: str,
    n_battles: int = 20,
) -> Dict:
    # Evaluate the model
    asyncio.set_event_loop(asyncio.new_event_loop())

    #model = DQNLightning(**model_kwargs)
    #model.load_from_checkpoint(model_path)
    if battle_format == "gen7randombattle":
        action_space = len(Gen7EnvSinglePlayer._ACTION_SPACE) - 1 # TODO why -1?
        #action_space = len(Gen7EnvSinglePlayer._ACTION_SPACE)
    
    model = DQN(
        #obs_size=obs_space,
        #n_actions=action_space,
        **model_kwargs)
    model.load_state_dict(model_state)
    results = evaluate_net(model, model_kwargs['stack_size'], battle_format, opponent_type, n_battles)
    print(results)
    return results

def bless(
    previous_best_model_state,
    previous_best_model_kwargs,
    model_state,
    model_kwargs,
    battle_format,
    n_battles: int = 5000,
):
    # Evaluate the model
    #model = DQNLightning(**model_kwargs)
    #model.load_from_checkpoint(model_path)
    #if battle_format == "gen7randombattle":
        #action_space = len(Gen7EnvSinglePlayer._ACTION_SPACE) - 1 # TODO why -1?
        #action_space = len(Gen7EnvSinglePlayer._ACTION_SPACE)
    asyncio.set_event_loop(asyncio.new_event_loop())
    
    model = DQN(
        **model_kwargs)
    model.load_state_dict(model_state)
    player = DQNPlayer(
        battle_format=battle_format, 
        net=model, 
        stack_size=model_kwargs['stack_size'], 
        max_concurrent_battles=0)

    opp_model = DQN(
        **previous_best_model_kwargs)
    opp_model.load_state_dict(previous_best_model_state)
    opponent = DQNPlayer(
        battle_format=battle_format, 
        net=opp_model, 
        stack_size=previous_best_model_kwargs['stack_size'], 
        max_concurrent_battles=0)
    asyncio.get_event_loop().run_until_complete(
        player.battle_against(opponent, n_battles=n_battles))
    # TODO another stage pitting max self versus probabilistic self

    results = dict(
        wins=player.n_won_battles,
        finished=player.n_finished_battles,
        ties=player.n_tied_battles,
        losses=player.n_lost_battles,
        win_rate=player.win_rate,
    )

    if player.win_rate > opponent.win_rate:
        best_model_state = model.state_dict()
        best_model_kwargs = model_kwargs
    else:
        best_model_state = previous_best_model_state
        best_model_kwargs = previous_best_model_kwargs

    return dict(
        results=results,
        blessed_model_state=best_model_state,
        blessed_model_kwargs=best_model_kwargs,
    )


# TODO use RL player instead of other kinds. can pull out action/obs space
# TODO is iterator frozen? or does it get new steps?