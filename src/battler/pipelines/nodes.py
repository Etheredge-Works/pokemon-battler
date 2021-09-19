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
    #opponent_name: str, 
    #opponent_id: int, 
    #model_type: str, model_name: str, model_id: int,
    lightning_kwargs: Dict,
    epochs: int,
    #replay_size: int,
    #warm_start_steps: int,
    #gamma: float,
    #eps_start: float,
    #eps_end: float,
    #eps_last_frame: int,
    #sync_rate: int,
    #lr: float,
    #episode_length: int,
    #batch_size: int,
    reward_kwargs: Dict,
    net_kwargs: Dict,
    #hidden_size: int,
    #learning_rate: float,
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
        #model=model,
        model_state=model.net.state_dict(),
        model_path=model_path
    )


def get_opponent(opponent_type: str):
    if opponent_type == 'random':
        return RandomOpponentPlayer
    elif opponent_type == 'self':
        return RLOpponentPlayer
    elif opponent_type == 'max_damage':
        return MaxDamagePlayer
    else:
        raise ValueError('Unknown opponent type: {}'.format(opponent_type))

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
        state = player.reset()
        done = False
        while not done:
            tensor_state = torch.tensor(state, dtype=torch.float32)
            y = model(tensor_state).item()
            probabilities = F.softmax(y)
            action = torch.multinomial(probabilities, 1).item()
            #values = model.forward(tensor_state)
            #action = values.argmax()

            #action = values.argmax()
            state, reward, done, _ = player.step(action)

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
        action_space = len(Gen7EnvSinglePlayer._ACTION_SPACE) -1
        #action_space = len(Gen7EnvSinglePlayer._ACTION_SPACE)
    
    model = DQN(
        obs_size=obs_space,
        n_actions=action_space,
        **model_kwargs)
    model.load_state_dict(model_state)
    player = DQNPlayer(battle_format=battle_format, net=model)
    #player = get_opponent(opponent_type)(battle_format=battle_format)
    opponent = get_opponent(opponent_type)(battle_format=battle_format)
    #results = pit_players(players=[player, opponent], n_challenges=20)
    #results = await cross_evaluate(players=[player, opponent], n_challenges=20))
    #results = asyncio.get_event_loop().run_until_complete(
        #cross_evaluate([player, opponent], n_challenges=20))
    
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


    #env_player = RLPlayer(battle_format=battle_format)
    #env_player.play_against(
        #env_algorithm=dqn_evaluation,
        #opponent=opponent,
        #env_algorithm_kwargs=dict(
            #model_state=model_state,
            #model_kwargs=model_kwargs,
        #)
    #)
    #return dict(
        #model=model,
        #trainer=trainer,
    #)