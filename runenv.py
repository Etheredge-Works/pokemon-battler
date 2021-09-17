from poke_env.player.env_player import Gen8EnvSinglePlayer, Gen7EnvSinglePlayer
from poke_env.environment.pokemon import Pokemon
from gym import spaces
from pytorch_lightning.core.hooks import ModelHooks
from torchdqn import *
from typing import Dict
from poke_env.environment.battle import Battle
import time

class SimpleRLPlayer(Gen7EnvSinglePlayer):
    def embed_stats(self, pokemon: Pokemon):
            return np.array([
                pokemon.base_stats['hp'],
                pokemon.base_stats['atk'],
                pokemon.base_stats['def'],
                pokemon.base_stats['spa'],
                pokemon.base_stats['spd'],
                pokemon.base_stats['spd'],
                #pokemon.base_stats['evasion'],
                #pokemon.base_stats['accuracy'],
            ]) / 180.

    def embed_boosts(self, pokemon: Pokemon):
            # TODO just get values? ordered dict?
            return np.array([
                pokemon.boosts['atk'],
                pokemon.boosts['def'],
                pokemon.boosts['spa'],
                pokemon.boosts['spd'],
                pokemon.boosts['spd'],
                pokemon.boosts['evasion'],
                pokemon.boosts['accuracy'],
            ]) / 6.

    def embed_moves(self, battle: Battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        moves_acc = np.ones(4)
        moves_other = np.zeros((4, 9))
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )
            moves_acc[i] = move.accuracy

            moves_other[i] = np.array([
                # bools
                move.breaks_protect,
                move.heal,
                move.is_protect_counter,
                move.is_protect_move,
                #move.is_recharge, not gen 7
                move.force_switch,

                # precent
                move.recoil,
                move.crit_ratio,

                # ints
                move.priority,
                #move.n_hit[0], move.n_hit[1],
                move.expected_hits
                #move.expected_hits,
                #move.self_boosts,
            ])
        return np.concatenate([
            moves_base_power, 
            moves_dmg_multiplier, 
            moves_acc,
            moves_other.flatten(),
            ])
        
    def embed_pokemon(self, pokemon):
        
        # TODO encode chance of move success? maybe use counter for accuracy of protected moves?
        return np.concatenate([
            #pok
            self.embed_stats(pokemon),
            self.embed_boosts(pokemon),
            np.array([pokemon.protect])
            [pokemon.level / 100.]
        ])

    def embed_battle(self, battle: Battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_vector = self.embed_moves(battle)

        # TODO : Add more embedding
        # TODO: gender 

        stats = [self.embed_stats(mon)for mon in battle.team.values()]
        stats = np.concatenate(stats)

        boosts = self.embed_boosts(battle.active_pokemon)
        opp_boosts = self.embed_boosts(battle.opponent_active_pokemon)

        hp = battle.active_pokemon.current_hp_fraction
        opp_hp = battle.opponent_active_pokemon.current_hp_fraction
        #opp_boosts = self.embed_boosts(battle.active_pokemon)
        # We count how many pokemons have not fainted in each team
        remaining_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        levels = np.array([mon.level for mon in battle.team.values()]) / 100.
        #opp_levels = [mon.level for mon in battle.active_pokemon]

        # Final vector with N components
        return np.concatenate([
            stats, # 6 * 6 = 36
            boosts, # 7
            opp_boosts, # 7
            levels, # 6
            moves_vector, # 12
            [hp, opp_hp], # 2
            [remaining_mon_team, remaining_mon_opponent]] # 2
        ) # 36 + 7 + 12 + 2 = 60
    # TODO add issue for swapping auto order for easier embedding
    # TODO running avarage of wins
    # TODO register models
    # TODO add tests on data

    # TODO if new model beats old model in 1_000_000 games, move to production
    # TODO hyperparam sweep with reward function changes too


    # TODO is height used for anything?
    # TODO add weight
    # TODO add height

    @property
    def observation_space(self) -> np.array:
        return spaces.Box(float("-inf"), float("inf"), shape=(108,))

    @property
    def action_space(self) -> List:
        """Returns the action space of the player. Must be implemented by subclasses."""
        return spaces.Discrete(18)
        #return spaces.Discrete(22)
        #return spaces.Discrete(self._ACTION_SPACE)

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value=2, # TODO change to 1 could cause 2 poke to sacrifice?
            hp_value=1,
            victory_value=30,
        )

from poke_env.player.random_player import RandomPlayer

def dqn_training(env, return_dict):
    torch.manual_seed(0)
    np.random.seed(0)
    #print(env)
    #print(env.action_space)
    #print(env.observation_space)
    model = DQNLightning(env)
    trainer = pl.Trainer()
    trainer = pl.Trainer(gpus=1, accelerator="dp", max_epochs=5)
    trainer.fit(model)

    # This call will finished eventual unfinshed battles before returning
    print("FINISHED")
    return_dict["model"] = model
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
    
class MaxDamagePlayer(RandomPlayer):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

if __name__ == "__main__":
    env_player = SimpleRLPlayer(battle_format="gen7randombattle")
    opponent = RandomPlayer(battle_format="gen7randombattle")
    opponent2 = MaxDamagePlayer(battle_format="gen7randombattle")

    '''
    # Training
    start = time.time()
    return_dict = {}
    env_player.play_against(
        env_algorithm=dqn_training,
        opponent=opponent,
        env_algorithm_kwargs=dict(return_dict=return_dict),
    )
    print("Training finished")
    mins, _ = divmod(time.time() - start, 60)
    print(f"train time: {mins}m")

    # Evaluation 1
    start = time.time()
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=opponent,
        env_algorithm_kwargs={'return_dict': return_dict},
    )
    mins, _ = divmod(time.time() - start, 60)
    print(f"random opp time: {mins}")
    # Evaluation 2
    start = time.time()
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=opponent2,
        env_algorithm_kwargs={'return_dict': return_dict},
    )
    mins, _ = divmod(time.time() - start, 60)
    print(f"smarter opp time: {mins}")
    # TODO investigate reseting twice
    # TODO add note about return type on action space and observation space


    '''
    ###############
 
    # Training
    return_dict = {}
    start = time.time()
    env_player.play_against(
        env_algorithm=dqn_training,
        opponent=opponent2,
        env_algorithm_kwargs=dict(return_dict=return_dict),
    )
    print("Training2 finished")
    mins, _ = divmod(time.time() - start, 60)
    print(f"train time: {mins}m")

    # Evaluation 1
    start = time.time()
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=opponent,
        env_algorithm_kwargs={'return_dict': return_dict},
    )
    mins, _ = divmod(time.time() - start, 60)
    print(f"random opp2 time: {mins}")
    # Evaluation 2
    start = time.time()
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=opponent2,
        env_algorithm_kwargs={'return_dict': return_dict},
    )
    mins, _ = divmod(time.time() - start, 60)
    print(f"smarter opp2 time: {mins}")
    # TODO investigate reseting twice
    # TODO add note about return type on action space and observation space



       