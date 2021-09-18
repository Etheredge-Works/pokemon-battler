from players.opponents import OpponentPlayer
from poke_env.player.env_player import Gen8EnvSinglePlayer, Gen7EnvSinglePlayer
from poke_env.environment.pokemon import Pokemon
from gym import spaces
from pytorch_lightning.core.hooks import ModelHooks
from torchdqn import *
from typing import Dict
from poke_env.environment.battle import Battle, AbstractBattle
import time
import utils
from utils import *


from poke_env.player.random_player import RandomPlayer
from players import SimpleRLPlayer, MaxDamagePlayer

def dqn_training(env, return_dict):
    torch.manual_seed(0)
    np.random.seed(0)
    #print(env)
    #print(env.action_space)
    #print(env.observation_space)
    #model = DQNLightning(env)
    #trainer.fit(model)
    model = return_dict["model"]
    model.build(env, return_dict["opponent"])
    trainer = return_dict["trainer"]
    trainer.fit(model)

    # This call will finished eventual unfinshed battles before returning
    print("FINISHED")
    #return_dict["model"] = model
    return_dict["trainer"] = trainer
    env.complete_current_battle()


def dqn_evaluation(player, return_dict, nb_episodes=1000):
    # Reset battle statistics
    player.reset_battles()
    trainer = return_dict["trainer"]
    model = return_dict["model"]
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

    #result = trainer.test(model)
    #print(result)
    #dqn.test(player, nb_episodes=nb_episodes, visualize=False, verbose=False)

    print(
        "DQN Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, player.n_finished_battles)
    )
    
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

    start = time.time()
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

