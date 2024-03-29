import numpy as np
from numpy.core.multiarray import concatenate
from poke_env.environment.move import Move

from poke_env.environment.pokemon import Pokemon
from poke_env.environment.battle import Battle
from icecream import ic


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

from poke_env.environment.move import Move

MOVE_DIM = 15
def embed_pokemon_moves(mon: Pokemon) -> np.ndarray:
    moves = np.zeros((4, MOVE_DIM))
    #ic(mon.moves)
    for i, m in enumerate(mon.moves.values()):
        if i > 3:
            #ic(mon.moves)
            pass
        else:
            moves[i] = embed_move(m)
        # TODO why does there sometimes get four moves?

    return moves

def embed_move(move: Move) -> np.ndarray:
    # -1 indicates that the move does not have a base power
    # or is not available
    # TODO move to embed pokemon
    if move is None:
        data = np.zeros(MOVE_DIM)
        #data[1] = -1
    else:
        does_boost = move.boosts is not None
        # TODO steals boosts
        data = np.array([
            move.type.value + 1, # EMBED TYPE
            move.base_power / 100.,
            move.accuracy,
            int(does_boost),
            # bools
            #int(move.breaks_protect),
            int(move.heal),
            #int(move.is_protect_counter),
            int(move.is_protect_move),
            #move.is_recharge, not gen 7
            int(move.force_switch),
            #int(move.ignore_ability),
            #int(move.ignore_evasion),
            #int(move.thaws_target),
            int(move.stalling_move),

            # precent
            move.recoil,
            move.crit_ratio,
            move.current_pp / 50.,
            #move.damage,
            move.drain,
            #int(move.ignore_immunity),
            move.max_pp / 50.,

            # ints
            move.priority,
            move.expected_hits
            #move.expected_hits,
            #move.self_boosts,
        ])
    return data
 

'''
def embed_moves(battle: Battle) -> np.ndarray:
    # -1 indicates that the move does not have a base power
    # or is not available
    # TODO move to embed pokemon
    moves_base_power = -np.ones(4)
    moves_dmg_multiplier = np.ones(4)
    moves_acc = np.ones(4)
    moves_other = np.zeros((4, 11))
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
            #int(move.breaks_protect),
            int(move.heal),
            #int(move.is_protect_counter),
            int(move.is_protect_move),
            #move.is_recharge, not gen 7
            int(move.force_switch),
            #int(move.ignore_ability),
            #int(move.ignore_evasion),
            #int(move.thaws_target),
            int(move.stalling_move),

            # precent
            move.recoil,
            move.crit_ratio,
            move.current_pp / 50.,
            #move.damage,
            move.drain,
            #int(move.ignore_immunity),
            move.max_pp / 50.,

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
'''
    
def embed_opponent_pokemons(battle: Battle) -> np.ndarray:
    pass

POKEMON_FLOAT_DIMS = 6+9
# TODO write up bug on typehint of weather
def embed_pokemon(pokemon: Pokemon) -> np.ndarray:
    # TODO encode chance of move success? maybe use counter for accuracy of protected moves?
    if pokemon is None:
        return np.zeros(POKEMON_FLOAT_DIMS)
    #embed_boosts(pokemon),
    return np.concatenate([
        #pok
        embed_stats(pokemon),
        #np.array([pokemon.protect]),
        np.array([
            float(pokemon.active),
            float(pokemon.current_hp / 250.) if pokemon.current_hp else 0.,
            float(pokemon.current_hp_fraction),
            float(pokemon.fainted),
            float(pokemon.first_turn),  # TODO remove
            float(pokemon.must_recharge),  # TODO remove
            #float(pokemon.preparing), TODO why tuple?
            float(pokemon.revealed),
            pokemon.status_counter / 6., # TODO some history here
            #float(pokemon.weight),
            pokemon.level / 100.
        ])
    ])


# TODO encode weather same as others? single stack weather yes. stackable should be one-hot
from poke_env.environment.pokemon_type import PokemonType
from battler.utils.encoders.abilities import encode_ability, encode_item, encode_pokemon
def encode_type(type: PokemonType) -> int:
    if type is None:
        return 0
    else:
        return type.value + 1

from poke_env.environment.weather import Weather
from poke_env.environment.status import Status

def encode_status(status: Status) -> int:
    if status is None:
        return 0
    return status.value + 1


POKEMON_ENUM_DIMS = 6
def enum_pokemon(battle: Battle, pokemon: Pokemon) -> np.ndarray:
    gen = gen_from_tag(battle.battle_tag)
    poke = encode_pokemon(battle.battle_tag, pokemon._species)
    type1 = encode_type(pokemon.type_1)
    type2 = encode_type(pokemon.type_2)
    
    ability = encode_ability(battle.battle_tag, pokemon.ability)
    item = encode_item(battle.battle_tag, pokemon.item)
    status = encode_status(pokemon.status)
    # TODO encode status
    
    return np.array([
        poke,
        status,
        type1,
        type2,
        ability,
        item,
    ])

# TODO renomoralize action spaces based on what moves are currenlty legal
from poke_env.environment.field import Field
def hot_field(battle) -> np.ndarray:
    encodings = np.zeros(len(Field))
    for idx, pos_field in enumerate(Field):
        if pos_field in battle.fields.keys():
            encodings[idx] = 1
    return encodings

from poke_env.environment.field import Field
from typing import Dict
def enum_weather(weathers: Dict[Weather, int]) -> np.ndarray:
    try:
        weather = list(weathers.keys())[0]
        return weather.value + 1
    except IndexError:
        return 0

def embed_field(battle: Battle):
    return np.array([
        int(battle.can_dynamax),
        int(battle.can_mega_evolve),
        #battle.can_zmove),
        # TODO bug sometimes is list int(battle.force_switch),
        int(battle.maybe_trapped),
        int(battle.opponent_can_dynamax),
        int(battle.trapped)
    ])


from typing import Dict
from poke_env.environment.side_condition import SideCondition
def hot_side(side_conditions: Dict[SideCondition, int]):
    conditions = np.zeros(len(SideCondition))
    for idx, condition in enumerate(SideCondition):
        if condition in side_conditions:
            conditions[idx] = side_conditions[condition]
            
    return conditions


import re
def gen_from_tag(tag):
    return re.search(r'gen(\d)', tag).group(1)


# TODO make embed functions a class with a method and callability
BATTLE_SHAPE = 1047
def embed_battle(battle: Battle) -> np.ndarray:
    #return BattleEmbed(battle)
    # -1 indicates that the move does not have a base power
    # or is not available
    poke_enums = np.array([enum_pokemon(battle, pokemon) for pokemon in battle.team.values()])
    poke_enums = np.swapaxes(np.array(poke_enums), 0, 1)

    opp_poke_enums = np.array([enum_pokemon(battle, pokemon) for pokemon in battle.opponent_team.values()])
    missing_axis = 6 - opp_poke_enums.shape[0]
    #opp_poke_enums = np.pad(opp_poke_enums, ((0, missing_axis), (0, 0)), 'constant')
    opp_poke_enums = np.concatenate([opp_poke_enums, np.zeros((missing_axis, 6))])
    opp_poke_enums = np.swapaxes(np.array(opp_poke_enums), 0, 1)

    ###
    field = embed_field(battle)
    hot_field_data = hot_field(battle)
    hot_side_conditions = hot_side(battle.side_conditions)
    opp_hot_side_conditions = hot_side(battle.opponent_side_conditions)
    weather = enum_weather(battle.weather)

    # lines up all the types and abilities and items
    ally_team = battle.team.values()
    #ally_team = sorted(
        #battle.team.values(), 
        # active first, fainted last, species next
        #key=lambda mon: (-1*int(mon.active),  int(mon.fainted), mon.species)
    #)
    poke_encodes = np.array([embed_pokemon(mon) for mon in ally_team])
    # TODO by putting active mon first, could also drop active indicator.
    # TODO this also frees up the next to think differnetly in differnt layers
    # TODO the other poke layers can think differently 

    opp_team = battle.opponent_team.values()
    #opp_team = sorted(
        ##battle.opponent_team.values(), 
        #key=lambda mon: (-1*int(mon.active),  int(mon.fainted), mon.species)
    #)

    opp_poke_encodes = np.array([embed_pokemon(mon) for mon in opp_team])
    missing_axis = 6 - opp_poke_encodes.shape[0]
    zeros = np.zeros((missing_axis, opp_poke_encodes.shape[1]))
    opp_poke_encodes = np.concatenate([opp_poke_encodes, zeros], axis=0)

    # TODO embed force switch so agent can make smart move in forced switch

    poke_moves = np.array([embed_pokemon_moves(pokemon) for pokemon in ally_team])
    #ic(poke_moves.shape)

    opp_poke_moves = np.array([embed_pokemon_moves(pokemon) for pokemon in opp_team])
    missing_axis = 6 - opp_poke_moves.shape[0]
    #ic(opp_poke_moves.shape)
    opp_poke_moves = np.concatenate([opp_poke_moves, np.zeros((missing_axis, *poke_moves.shape[1:]))]) #TODO is -1 better?
    #ic(opp_poke_moves.shape)

    # TODO : Add more embedding
    # TODO: gender 
    # TODO is plays the sllooowwest game. it needs self play to get out of rut

    # TODO maybe encode turn count?

    boosts = embed_boosts(battle.active_pokemon)
    opp_boosts = embed_boosts(battle.opponent_active_pokemon)

    # We count how many pokemons have not fainted in each team
    remaining_mon_team = len([mon for mon in battle.team.values() if not mon.fainted]) / 6
    remaining_mon_opponent = (
        len([mon for mon in battle.opponent_team.values() if not mon.fainted]) / 6
    )
    # TODO should dead pokes be ignored? is there anything to gain form thier info?
    # TODO attention to only active pokes?

    #levels = np.array([mon.level for mon in battle.team.values()]) / 100.
    #opp_levels = [mon.level for mon in battle.active_pokemon]

    # TODO how is confusion handled???
    # Final vector with N components
    assert poke_enums.shape == opp_poke_enums.shape, f"{poke_enums.shape} - {opp_poke_enums.shape}"
    # poke_encodes.shape == opp_poke_encodes.shape, f"{poke_encodes.shape} - {opp_poke_encodes.shape}"
    embedding = np.concatenate([
            # to embed
            [weather],
            poke_enums.flatten(),
            poke_encodes.flatten(),
            
            opp_poke_enums.flatten(),
            opp_poke_encodes.flatten(),

            poke_moves.flatten(),
            opp_poke_moves.flatten(),

            field,
            hot_field_data,
            hot_side_conditions,
            opp_hot_side_conditions,

            #stats, # 6 * 6 = 36
            boosts, # 7
            opp_boosts, # 7
            #levels, # 6
            #moves_vector, # 12
            #mons_emb,
            #opp_mons_emb,
            [
                remaining_mon_team, 
                remaining_mon_opponent
            ],
            #[hp, opp_hp], # 2
            #[remaining_mon_team, remaining_mon_opponent]] # 2
    ]) # 36 + 7 + 12 + 2 = 60
    #ic()
    #ic(embedding)
    #ic(embedding.shape)
    assert embedding.shape[0] == BATTLE_SHAPE, f"{embedding.shape} != {BATTLE_SHAPE}"
    return embedding
