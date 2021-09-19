from kedro.pipeline import Pipeline, node
from pytorch_lightning.core import lightning

from .nodes import (
    evaluate,
    train
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
                reward_kwargs="params:reward_kwargs",
                net_kwargs="params:dqn_kwargs",
            ),
            outputs=dict(
                model_path="dqn_model_path",
                model_state="dqn_model_state"
            ),
        ),
    ])
    eval_p = Pipeline([
        node(
            evaluate,
            inputs=dict(
                model_state="dqn_model_state",
                model_kwargs="params:dqn_kwargs",
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
                model_kwargs="params:dqn_kwargs",
                n_battles="params:n_battles",
                obs_space="params:obs_space",
                battle_format="params:battle_format",
                opponent_type="params:max_opponent",
            ),
            outputs='dqn_max_results'
        ),
    ])
    return train_p, eval_p