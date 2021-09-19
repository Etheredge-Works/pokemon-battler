"""
Helper functions for working with poke_env
"""
import torch
import numpy as np


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

    print("FINISHED")
    #return_dict["model"] = model
    return_dict["trainer"] = trainer
    # This call will finished eventual unfinshed battles before returning
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
    