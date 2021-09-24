from kedro.pipeline import Pipeline, node
from pytorch_lightning.core import lightning

from .nodes import (
    evaluate,
    train,
    bless,
)
#from .analysis import exploration

#from ..utils import split_data


def create_data_pipeline(**kwargs):
    train_p = Pipeline([
        node(
            train, 
            inputs=dict(
                battle_format="params:battle_format",
                opponent_type="params:train_opponent",
                lightning_kwargs="params:lightning_kwargs",
                epochs="params:epochs",
                model_path="params:checkpoint_path",
                reward_kwargs="params:rl_player_kwargs",
                net_kwargs="params:dqn_kwargs",
            ),
            outputs=dict(
                #model_path="dqn_model_path",
                best_model_state="dqn_model_state",
                best_model_kwargs="dqn_model_kwargs",
            ),
        ),
    ])
    # TODO print summary of net and training
    eval_p = Pipeline([
        node(
            evaluate,
            inputs=dict(
                model_state="dqn_model_state",
                model_kwargs="dqn_model_kwargs",
                n_battles="params:n_battles",
                obs_space="params:obs_space",
                battle_format="params:battle_format",
                opponent_type="params:random_opponent",
            ),
            outputs='dqn_random_results'
        ),
        node(
            evaluate,
            inputs=dict(
                model_state="dqn_model_state",
                model_kwargs="dqn_model_kwargs",
                n_battles="params:n_battles",
                obs_space="params:obs_space",
                battle_format="params:battle_format",
                opponent_type="params:max_opponent",
            ),
            outputs='dqn_max_results'
        ),
        # best_model
    ])
    bless_p = Pipeline([
        node(
            bless,
            inputs=dict(
                previous_best_model_state="previous_best_dqn_model_state",
                previous_best_model_kwargs="previous_best_dqn_model_kwargs",
                model_state="dqn_model_state",
                model_kwargs="dqn_model_kwargs",
                n_battles="params:blessing_n_battles",
                battle_format="params:battle_format",
            ),
            outputs=dict(
                blessed_model_state='best_dqn_model_state',
                blessed_model_kwargs='best_dqn_model_kwargs',
                results='blessed_dqn_results'
            )
        )
    ])
    return train_p, eval_p, bless_p