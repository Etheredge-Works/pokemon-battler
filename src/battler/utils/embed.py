import numpy as np
from poke_env.environment.move import Move

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
    # TODO move to embed pokemon
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

        '''
        moves_other[i] = np.array([
            # bools
            int(move.breaks_protect),
            int(move.heal),
            int(move.is_protect_counter),
            int(move.is_protect_move),
            #move.is_recharge, not gen 7
            int(move.force_switch),
            int(move.ignore_ability),
            int(move.ignore_evasion),
            int(move.thaws_target),
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
        '''
    return np.concatenate([
        moves_base_power, 
        moves_dmg_multiplier, 
        moves_acc,
        #moves_other.flatten(),
        ])

    
def embed_opponent_pokemons(battle: Battle) -> np.ndarray:
    pass


def embed_pokemon(pokemon: Pokemon) -> np.ndarray:
    # TODO encode chance of move success? maybe use counter for accuracy of protected moves?
    if pokemon is None:
        return np.zeros(6)
    #embed_boosts(pokemon),
    return np.concatenate([
        #pok
        embed_stats(pokemon),
        #np.array([pokemon.protect]),
        np.array([
            float(pokemon.active),
            float(pokemon.current_hp / 250.),
            float(pokemon.current_hp_fraction),
            float(pokemon.fainted),
            float(pokemon.first_turn),
            float(pokemon.must_recharge),
            #float(pokemon.preparing), TODO why tuple?
            float(pokemon.revealed),
            #float(pokemon.weight),
            pokemon.level / 100.
        ])
    ])

'''
import pandas as pd
ITEMS = {}
gen7_items_df = pd.read_csv('gen7/items.csv')
ITEMS = {name: value for value, name in gen7_items_df.values}
TYPES = {name: value for value, name in Pokemon.TYPES.items()}
# TODO encoding moves would require me to have memory of what moves opponents used last turn
# TODO encode moves used last turn
#POKEMON = {name: value for value, name in pd.read_csv('gen7/pokemon.csv')}
# TODO needed for loading csv
#with open(
    #os.path.join(
        #os.path.dirname(os.path.realpath(__file__)),
        #"data",
        #"pokedex_by_gen",
        #"gen7_pokedex.json",
    #)
# TODO need ivs/evs to be in embed due to things like hidden power
# TODO will need level
class MoveEmbedding:
    def __init__(self, move: Move):
        self.values = np.array([
            move.base_power / 100.,
            move.accuracy,
            #move.boosts['atk'],
        ])
        self.type = TYPES[move.type]
        self.move

class PokemonEmbed:
    def __init__(self, pokemon: Pokemon):
        self.values = embed_pokemon(pokemon)
        self.item = ITEMS[pokemon.item]
        #self.moves = [MoveEmbed(move) for move in pokemon.moves]
        self.type_1 = int(pokemon.type_1)
        self.type_2 = int(pokemon.type_2)
        self.species = POKEMON[pokemon.species]
        # TODO maybe encode name and types?
        #self.ability = int(pokemon.ability)

        #self.type = 
        #self.nature = embed_nature(pokemon.nature)
class BattleEmbedding:
    def __init__(self, battle: Battle):
        #self.battle = battle
        self.team = [PokemonEmbed(mon) for mon in battle.team]
        #self.opp_team = [PokemonEmbed(mon) for mon in battle.team]
        self.opponent_pokemon = embed_pokemon(battle.opponent_active_pokemon)
        self.moves = embed_moves(battle)
        self.opponent_moves = embed_moves(battle)

    
'''
# TODO encode weather same as others? single stack weather yes. stackable should be one-hot
from poke_env.environment.pokemon_type import PokemonType
from battler.utils.encoders.abilities import encode_ability, encode_item
def encode_type(type: PokemonType) -> int:
    if type is None:
        return 0
    else:
        return type.value + 1

def enum_pokemon(battle, pokemon: Pokemon) -> np.ndarray:
    type1 = encode_type(pokemon.type_1)
    type2 = encode_type(pokemon.type_2)
    
    ability = encode_ability(battle._format, pokemon.ability)
    item = encode_item(battle._format, pokemon.item)
    
    return np.array([
        type1,
        type2,
        ability,
        item,
    ])

class BattleEmbed:
    def __init__(self, battle: Battle) -> None:
        self.type1s = [PokemonType(pokemon.type_1) for pokemon in battle.team.values()]
        self.type2s = [
            PokemonType(pokemon.type_2) 
            if pokemon.type_2 is not None else -1
            for pokemon in battle.team.values()]
        self.abilities = [encode_ability(battle._format, pokemon.ability) for pokemon in battle.team.values()]
        self.items = [encode_item(battle._format, pokemon.item) for pokemon in battle.team.values()]

        moves_vector = embed_moves(battle)

        stats = [embed_stats(mon)for mon in battle.team.values()]
        stats = np.concatenate(stats)
        # TODO maybe encode turn count?

        boosts = embed_boosts(battle.active_pokemon)
        opp_boosts = embed_boosts(battle.opponent_active_pokemon)
        #mons_emb = np.concatenate([embed_pokemon(mon) for mon in battle.team.values()])
        #opp_mons_emb = np.concatenate([embed_pokemon(mon) for mon in battle.opponent_team.values()])

        opp_hp = battle.opponent_active_pokemon.current_hp_fraction
        opp_level = battle.opponent_active_pokemon.level / 100.
        remaining_mon_team = len([mon for mon in battle.team.values() if not mon.fainted]) / 6
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if not mon.fainted]) / 6
        )


        # Final vector with N components
        self.others = np.concatenate([
                stats, # 6 * 6 = 36
                boosts, # 7
                opp_boosts, # 7
                moves_vector, # 12
                [
                    opp_level,
                    opp_hp, 
                    remaining_mon_team, 
                    remaining_mon_opponent
                ],
        ])

    def tensor(self):
        return self.type1s, self.type2s, self.abilities, self.items, self.others

def embed_battle(battle: Battle) -> np.ndarray:
    #return BattleEmbed(battle)
    # -1 indicates that the move does not have a base power
    # or is not available
    embeddings = enum_pokemon(battle, battle.active_pokemon)
    moves_vector = embed_moves(battle)

    # TODO : Add more embedding
    # TODO: gender 

    stats = [embed_stats(mon)for mon in battle.team.values()]
    stats = np.concatenate(stats)
    # TODO maybe encode turn count?

    boosts = embed_boosts(battle.active_pokemon)
    opp_boosts = embed_boosts(battle.opponent_active_pokemon)
    #mons_emb = np.concatenate([embed_pokemon(mon) for mon in battle.team.values()])
    #opp_mons_emb = np.concatenate([embed_pokemon(mon) for mon in battle.opponent_team.values()])

    #hp = battle.active_pokemon.current_hp_fraction
    opp_hp = battle.opponent_active_pokemon.current_hp_fraction
    opp_level = battle.opponent_active_pokemon.level / 100.
    #opp_boosts = embed_boosts(battle.active_pokemon)
    # We count how many pokemons have not fainted in each team
    remaining_mon_team = len([mon for mon in battle.team.values() if not mon.fainted]) / 6
    remaining_mon_opponent = (
        len([mon for mon in battle.opponent_team.values() if not mon.fainted]) / 6
    )

    #levels = np.array([mon.level for mon in battle.team.values()]) / 100.
    #opp_levels = [mon.level for mon in battle.active_pokemon]

    # Final vector with N components
    return np.concatenate([
            # to embed
            embeddings,

            stats, # 6 * 6 = 36
            boosts, # 7
            opp_boosts, # 7
            #levels, # 6
            moves_vector, # 12
            #mons_emb,
            #opp_mons_emb,
            [
                opp_level,
                opp_hp, 
                remaining_mon_team, 
                remaining_mon_opponent
            ],
            #[hp, opp_hp], # 2
            #[remaining_mon_team, remaining_mon_opponent]] # 2
    ]) # 36 + 7 + 12 + 2 = 60
