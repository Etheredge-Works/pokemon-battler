def dqn_evaluation(player, dqn, nb_episodes):
    # Reset battle statistics
    player.reset_battles()
    dqn.test(player, nb_episodes=nb_episodes, visualize=False, verbose=False)

    print(
        "DQN Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, nb_episodes)
    )

def eval(player, nb_episodes):
    # Reset battle statistics
    player.reset_battles()
    player.test(nb_episodes=nb_episodes, visualize=False, verbose=False)

    print(
        "Random Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, nb_episodes)
    )


from poke_env.player import RandomPlayer

def foo(env_player):
    # Ths code of MaxDamagePlayer is not reproduced for brevity and legibility
    # It can be found in the complete code linked above, or in the max damage example
    #second_opponent = MaxDamagePlayer(battle_format="gen8randombattle")
    #second_

    # Evaluation
    print("Results against random player:")
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": 100},
    )

    print("\nResults against max player:")
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=second_opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": 100},
    )
