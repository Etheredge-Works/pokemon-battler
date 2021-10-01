from kedro.pipeline import Pipeline, node
from pytorch_lightning.core import lightning

from .nodes import (
    train,
)
#from .analysis import exploration

#from ..utils import split_data


def create_pipeline(**kwargs):
    train_p = Pipeline([
        node(
            train, 
            inputs=dict(
                #opponent_type="params:train_opponent",
                lightning_kwargs="params:ppo_lightning_kwargs",
                epochs="params:epochs",
            ),
            outputs=None,
            #outputs=dict(
                #model_path="dqn_model_path",
                #best_model_state="dqn_model_state",
                #best_model_kwargs="dqn_model_kwargs",
            #),
        ),
    ])
    return train_p