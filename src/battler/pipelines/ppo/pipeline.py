from kedro.pipeline import Pipeline, node
from pytorch_lightning.core import lightning

from .nodes import train

def create_pipeline(**kwargs):
    train_p = Pipeline([
        node(
            train, 
            inputs=dict(
                lightning_kwargs="params:ppo_lightning_kwargs",
                trainer_kwargs="params:trainer_kwargs",
            ),
            outputs=None,
        ),
    ])
    return train_p