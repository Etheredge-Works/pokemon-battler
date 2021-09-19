from kedro.pipeline import Pipeline, node

from .nodes import (
    train
)
#from .analysis import exploration

#from ..utils import split_data


def create_data_pipeline(**kwargs):
    return Pipeline([
        node(
            train, 
            inputs=dict(
                battle_format="params:battle_format",
                opponent_type="params:opponent_type",
                epochs="params:epochs",
                replay_size="params:replay_size",
                warm_start_steps="params:warm_start_steps",
                gamma="params:gamma",
                eps_start="params:eps_start",
                eps_end="params:eps_end",
                eps_last_frame="params:eps_last_frame",
                sync_rate="params:sync_rate",
                lr="params:lr",
                episode_length="params:episode_length",
                batch_size="params:batch_size",
                reward_kwargs="params:reward_kwargs",
                net_kwargs="params:dqn_kwargs",
            ),
            outputs=dict(
                model="raw_kddcup99_data",
                targets="raw_kddcup99_targets"
            ),
        ),
    ])