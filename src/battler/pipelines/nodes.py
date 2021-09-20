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
from battler.players.players import RandomOpponentPlayer
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


    opp = get_opponent(opponent_type)(battle_format=battle_format)
    # Train the model
    trainer = Trainer(
        gpus=1,
        accelerator='dp',
        max_epochs=epochs,
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

    # Return the model
    return dict(
        model_state=model.net.state_dict(),
        model_path=model_path
    )


from battler.utils import get_opponent

def dqn_evaluation(player, model_kwargs, model_state, nb_episodes=1000):
    # Reset battle statistics
    player.reset_battles()

    model = DQN(
        obs_size=player.observation_space.shape[0],
        n_actions=player.action_space.n,
        **model_kwargs)
    model.load_state_dict(model_state)
    model.eval()

    import torch.nn.functional as F
    print("Evaluating model")
    for _ in range(nb_episodes):
        # Reset the environment
        single_state = player.reset()
        state = deque([single_state for _ in range(model_kwargs['stack_size'])], maxlen=model_kwargs['stack_size'])
        done = False
        while not done:
            tensor_state = torch.tensor(state, dtype=torch.float32)
            y = model(tensor_state).item()
            probabilities = F.softmax(y)
            action = torch.multinomial(probabilities, 1).item()

            single_state, reward, done, _ = player.step(action)
            state.append(single_state)

    print(
        "DQN Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, player.n_finished_battles)
    )

from poke_env.player.env_player import Gen7EnvSinglePlayer
from poke_env.player.utils import cross_evaluate
from battler.deploy.dqn_player import DQNPlayer
async def pit_players(players, n_challenges=20):
    cross_evaluation = await cross_evaluate(players, n_challenges=20)
    return cross_evaluation

def evaluate_net(
    model,
    stack_size,
    battle_format,
    opponent_type,
    n_battles,
):
    player = DQNPlayer(battle_format=battle_format, net=model, stack_size=stack_size, max_concurrent_battles=0)
    opponent = get_opponent(opponent_type)(battle_format=battle_format, max_concurrent_battles=0)

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

    #model = DQNLightning(**model_kwargs)
    #model.load_from_checkpoint(model_path)
    if battle_format == "gen7randombattle":
        action_space = len(Gen7EnvSinglePlayer._ACTION_SPACE) - 1 # TODO why -1?
        #action_space = len(Gen7EnvSinglePlayer._ACTION_SPACE)
    
    model = DQN(
        obs_size=obs_space,
        n_actions=action_space,
        **model_kwargs)
    model.load_state_dict(model_state)
    results = evaluate_net(model, model_kwargs['stack_size'], battle_format, opponent_type, n_battles)
    print(results)
    return results

# TODO use RL player instead of other kinds. can pull out action/obs space
# TODO is iterator frozen? or does it get new steps?