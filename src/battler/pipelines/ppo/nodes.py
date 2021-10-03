from typing import Dict

from poke_env.player.random_player import RandomPlayer
from battler.models.ppo import PPOLightning
#from battler.players.players import RLPlayer, MaxDamagePlayer
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import random
import string
from poke_env.player_configuration import PlayerConfiguration
import pytorch_lightning as pl

# TODO use attention and pad moves
# TODO trainer kwargs
def train(
    lightning_kwargs: Dict,
    trainer_kwargs: Dict,
) -> Dict:

    # TODO change eval to challange existing players by id
    pl.seed_everything(4)

    trainer = Trainer(
        **trainer_kwargs,
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