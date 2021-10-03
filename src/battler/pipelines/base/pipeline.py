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
            inputs=None,
            outputs="pls",
        ),
    ])
    return train_p