from typing import Dict

from poke_env.player.random_player import RandomPlayer
from battler.models.ppo import PPOLightning
from battler.players.players import RLPlayer, MaxDamagePlayer
from pytorch_lightning import Trainer
import pytorch_lightning as pl


def train_wrapper(
    env, opponent, other
):
    trainer = Trainer(
        gpus=1,
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
    model = PPOLightning(env)
    trainer.fit(model)
    other['model'] = model
    other['trainer'] = trainer

# TODO use attention and pad moves
# TODO trainer kwargs
def train(
    battle_format: str,
    #opponent_type: str, 
    lightning_kwargs: Dict,
    epochs: int,
    reward_kwargs: Dict,
    net_kwargs: Dict,
    model_path: str,
) -> Dict:

    opp = MaxDamagePlayer(battle_format=battle_format)
    #opp = RandomPlayer(battle_format=battle_format)
    train_kwargs = dict(
        #model=None,
        #trainer=None,
        opponent=opp,
        other = {}
    )

    # TODO change eval to challange existing players by id
    env_player = RLPlayer(
        battle_format=battle_format, 
        reward_kwargs=reward_kwargs,
        obs_space=lightning_kwargs['obs_space'],
        stack_size=lightning_kwargs['stack_size'],)

    env_player.play_against(
        env_algorithm=train_wrapper,
        opponent=opp,
        env_algorithm_kwargs=train_kwargs
    )

    #trainer.save_checkpoint(model_path)
    #pr.disable()
    #s = io.StringIO.StringIO()
    #ps = pstats.Stats(pr, stream=s)
    #ps.dump_stats('/tmp/stats.dmp')

    # Return the model
    #return dict(
        #best_model_state=model.best_net.state_dict(),
        #best_model_kwargs=model.best_net.get_kwargs(),
        #model_state=model.net.state_dict(),
        #model_kwargs=model.net.get_kwargs(),
        #model_path=model_path
    #)