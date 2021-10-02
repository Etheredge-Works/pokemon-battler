from typing import Dict

from poke_env.player.random_player import RandomPlayer
from battler.models.ppo import PPOLightning
from battler.players.players import RLPlayer, MaxDamagePlayer
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import random
import string
from poke_env.player_configuration import PlayerConfiguration
import pytorch_lightning as pl


def train_wrapper(
    env_name, other, kwargs, env
):
    trainer = Trainer(
        gpus=0,
        accelerator='dp',
        max_epochs=1000,
        #log_every_n_steps=1,
        flush_logs_every_n_steps=4_000,
        logger=pl.loggers.MLFlowLogger(
            experiment_name='ppo_lightning',
        )
        #logger=pl.loggers.TensorBoardLogger(
            #'.',
            #log_graph=True,
            #default_hp_metric=False,
            #save_dir='tb_logs',
            #name='ppo_logs',
        #)
    )
    model = PPOLightning(env_name, env_kwargs=kwargs, raw_env=env)
    trainer.fit(model)
    return model

from poke_env.player.env_player import DummyPlayer
# TODO use attention and pad moves
# TODO trainer kwargs
def train(
    lightning_kwargs: Dict,
    epochs: int,
) -> Dict:

    # TODO change eval to challange existing players by id
    pl.seed_everything(4)

    trainer = Trainer(
        gpus=1,
        accelerator='dp',
        max_epochs=epochs,
        #log_every_n_steps=1,
        flush_logs_every_n_steps=4_000,
        logger=pl.loggers.MLFlowLogger(
            experiment_name='ppo_lightning',
        )
    )
    model = PPOLightning(**lightning_kwargs)
    trainer.fit(model)

    return model


def blesser():
    """
    Gets all staged models and the current production model and has them duke it out
    best one gets moved to deployment. The rest go archive.
    """