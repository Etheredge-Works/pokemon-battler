import numpy as np

from poke_env.environment.pokemon import Pokemon
from poke_env.environment.battle import Battle


def embed_stats(pokemon: Pokemon) -> np.ndarray:
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

def embed_boosts(pokemon: Pokemon) -> np.ndarray:
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


def embed_moves(battle: Battle) -> np.ndarray:
    # -1 indicates that the move does not have a base power
    # or is not available
    moves_base_power = -np.ones(4)
    moves_dmg_multiplier = np.ones(4)
    moves_acc = np.ones(4)
    moves_other = np.zeros((4, 16))
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
            int(move.is_protect_counter),
            int(move.is_protect_move),
            #move.is_recharge, not gen 7
            int(move.force_switch),

            # precent
            move.recoil,
            move.crit_ratio,
            move.current_pp / 50.,
            #move.damage,
            move.drain,
            int(move.ignore_ability),
            int(move.ignore_evasion),
            #int(move.ignore_immunity),
            move.max_pp / 50.,
            move.stalling_move,
            int(move.thaws_target),



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

    
def embed_pokemon(pokemon: Pokemon) -> np.ndarray:
    # TODO encode chance of move success? maybe use counter for accuracy of protected moves?
    return np.concatenate([
        #pok
        embed_stats(pokemon),
        embed_boosts(pokemon),
        np.array([pokemon.protect]),
        np.array([
            float(pokemon.active),
            float(pokemon.current_hp / 250.),
            float(pokemon.current_hp_fraction),
            float(pokemon.fainted),
            float(pokemon.first_turn),
            float(pokemon.must_recharge),
            float(pokemon.preparing),
            float(pokemon.revealed),
            float(pokemon.weight),
            pokemon.level / 100.
        ])
    ])


def embed_battle(battle: Battle) -> np.ndarray:
    # -1 indicates that the move does not have a base power
    # or is not available
    moves_vector = embed_moves(battle)

    # TODO : Add more embedding
    # TODO: gender 

    stats = [embed_stats(mon)for mon in battle.team.values()]
    stats = np.concatenate(stats)

    boosts = embed_boosts(battle.active_pokemon)
    opp_boosts = embed_boosts(battle.opponent_active_pokemon)

    hp = battle.active_pokemon.current_hp_fraction
    opp_hp = battle.opponent_active_pokemon.current_hp_fraction
    #opp_boosts = embed_boosts(battle.active_pokemon)
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
