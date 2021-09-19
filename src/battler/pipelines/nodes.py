from poke_env.player.random_player import RandomPlayer
from pytorch_lightning.accelerators import accelerator
from typing import Dict
import time
from pytorch_lightning import Trainer
import torch
import numpy as np

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

def dqn_evaluation(player, model, trainer, nb_episodes=1000):
    # Reset battle statistics
    player.reset_battles()

    print("Evaluating model")
    for _ in range(nb_episodes):
        # Reset the environment
        state = player.reset()
        done = False
        while not done:
            tensor_state = torch.tensor(state, dtype=torch.float32)
            values = model.forward(tensor_state)
            action = values.argmax()
            state, reward, done, _ = player.step(action)

    print(
        "DQN Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, player.n_finished_battles)
    )

def train(
    battle_format: str,
    opponent_type: str, 
    #opponent_name: str, 
    #opponent_id: int, 
    #model_type: str, model_name: str, model_id: int,
    epochs: int,
    replay_size: int,
    warm_start_steps: int,
    gamma: float,
    eps_start: float,
    eps_end: float,
    eps_last_frame: int,
    sync_rate: int,
    lr: float,
    episode_length: int,
    batch_size: int,
    reward_kwargs: Dict,
    net_kwargs: Dict,
    #hidden_size: int,
    #learning_rate: float,
) -> Dict:

    if opponent_type == 'random':
        opp = RandomOpponentPlayer(battle_format=battle_format)
    elif opponent_type == 'self':
        opp = RLOpponentPlayer(battle_format=battle_format)
    else:
        raise ValueError('Unknown opponent type: {}'.format(opponent_type))

    #opp = RandomPlayer(battle_format=battle_format)
    # Train the model
    trainer = Trainer(
        gpus=1,
        accelerator='dp',
        max_epochs=epochs,
    )
    model = DQNLightning(
        replay_size=replay_size,
        warm_start_steps=warm_start_steps,
        gamma=gamma,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_last_frame=eps_last_frame,
        sync_rate=sync_rate,
        lr=lr,
        episode_length=episode_length,
        batch_size=batch_size,
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

    # Return the model
    return dict(
        model=model,
        trainer=trainer,
    )


'''
if __name__ == "__main__":
    env_player = SimpleRLPlayer(battle_format="gen7randombattle")
    opponent = RandomPlayer(battle_format="gen7randombattle")
    opponent2 = MaxDamagePlayer(battle_format="gen7randombattle")

    ###############
 
    # Training
    trainer = pl.Trainer(gpus=1, accelerator="dp", max_epochs=10)
    model = DQNLightning()
    return_dict = dict(
        trainer=trainer,
        model=model,
        opponent=OpponentPlayer(battle_format="gen7randombattle"),
    )

    env_player.play_against(
        env_algorithm=dqn_training,
        opponent=opponent2,
        env_algorithm_kwargs=dict(return_dict=return_dict),
    )
    print("Training2 finished")
    mins = (time.time() - start) / 60
    print(f"train time: {mins}m")
    # Evaluation 1
    start = time.time()
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=opponent,
        env_algorithm_kwargs={'return_dict': return_dict},
    )
    mins = (time.time() - start) / 60
    print(f"random opp2 time: {mins}")
    # Evaluation 2
    start = time.time()
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=opponent2,
        env_algorithm_kwargs={'return_dict': return_dict},
    )
    mins = (time.time() - start) / 60
    print(f"smarter opp2 time: {mins}")
    # TODO investigate reseting twice
    # TODO add note about return type on action space and observation space


    # Create the environment
    #battle = Battle(opponent_type, opponent_name, opponent_id)
    #env = Gen8EnvSinglePlayer(battle, opponent_type, opponent_name, opponent_id)
    #env.seed(opponent_id)

    # Create the model
    #model = DQN(
        #env.observation_space,
        #env.action_space,
        #hidden_size,
        #learning_rate,
        #gamma,
        #epsilon_start,
    #)

    # Create the trainer
    #trainer = Trainer(
        #model,
        #env,
        #train_steps,
        #batch_size,
        #ModelHooks(
            #model_type, model_name, model_id,
            #train_steps,
            #hidden_size,
            #learning_rate,
            #gamma,
            #epsilon_start,
        #),
    #)


'''