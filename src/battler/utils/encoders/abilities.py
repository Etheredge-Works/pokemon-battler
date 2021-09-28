import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import poke_env
import json


DATA_PATH = os.path.join(
    os.path.dirname(os.path.realpath(poke_env.__file__)),
    "data")

def sanitize_ability(ability: str) -> str:
    return ability.lower()\
                .replace(" ", "")\
                .replace("'", "")\
                .replace("-", "")\
                .replace("(", "")\
                .replace(")", "")

def create_ability_encoder(gen: str) -> dict:
    file_path = os.path.join(
        DATA_PATH,
        "pokedex_by_gen",
        f"{gen}_pokedex.json")

    with open(file_path, 'r') as f:
        data = json.load(f)
    
    ability_names = set()
    for pokemon_name, pokemon_values in data.items():
        for ability_id, ability_name in pokemon_values['abilities'].items():
            ability_names.add(sanitize_ability(ability_name))

    le = LabelEncoder()
    print(f"ability_count: {len(ability_names)}")
    le.fit(list(ability_names))
    return le
    

GEN7_ABILITY_ENCODER = create_ability_encoder("gen7")
GEN8_ABILITY_ENCODER = create_ability_encoder("gen8")
GEN8_ABILITIES = GEN8_ABILITY_ENCODER.classes_
# TODO refine for gen 8


def encode_ability(battle_format: str, ability: str) -> int:
    if ability is None or ability == '':
        return 0
    ability = sanitize_ability(ability)
    return GEN8_ABILITY_ENCODER.transform([ability])[0] + 1
    if 'gen7' in battle_format:
        return GEN7_ABILITY_ENCODER.transform([ability])[0]
    else:
        raise ValueError("Invalid battle format")


def create_encode_ability(battle_format: str) -> int:
    if 'gen7' in battle_format:
        return GEN7_ABILITY_ENCODER
    else:
        raise ValueError("Invalid battle format")

###################
def clean_name(name: str) -> str:
    return name\
        .lower()\
        .replace(" ", "")\
        .replace("'", "")\
        .replace(".", "")\
        .replace("-", "")\
        .replace("(", "")\
        .replace(")", "")

def create_pokemon_encoder(gen: str) -> dict:
    file_path = os.path.join(
        DATA_PATH,
        "pokedex_by_gen",
        f"{gen}_pokedex.json")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # TODO formeOrder?o
    pokemon_names = set()
    for pokemon_name, pokemon_values in data.items():
        pokemon_names.add(clean_name(pokemon_name))
        if 'formeOrder' in pokemon_values:
            for name in pokemon_values.get('formeOrder'):
                pokemon_names.add(clean_name(name))
    
    print(f"pokemon count : {len(pokemon_names)}")
    le = LabelEncoder()
    le.fit(list(pokemon_names))
    return le

GEN8_POKEMON_ENCODER = create_pokemon_encoder("gen8")
GEN8_POKEMON = GEN8_POKEMON_ENCODER.classes_

def encode_pokemon(battle_format: str, pokemon: str) -> int:
    if pokemon is None or pokemon == '':
        return 0
    return GEN8_POKEMON_ENCODER.transform([clean_name(pokemon)])[0] + 1

    if 'gen7' in battle_format:
        return GEN7_POKEMON_ENCODER.transform([pokemon])[0]
    else:
        raise ValueError("Invalid battle format")

###################
def create_item_encoder(gen: str) -> dict:
    df = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "data",
            gen,
            "items.csv"
        )
    )
    item_names = df["item"].values
    item_names = [name.lower() for name in item_names]
    item_names = [name.replace(" ", "") for name in item_names]

    print(f"item_count: {len(item_names)}")
    le = LabelEncoder()
    le.fit(item_names)
    return le

GEN7_ITEM_ENCODER = create_item_encoder("gen7")
GEN8_ITEM_ENCODER = create_item_encoder("gen8")

def encode_item(battle_format: str, item: str) -> int:
    if item is None or item == '':
        return 0 
    return GEN8_ITEM_ENCODER.transform([item])[0] + 1

    if 'gen7' in battle_format:
        return GEN7_ITEM_ENCODER.transform([item])[0]
    else:
        raise ValueError("Invalid battle format")

###################

# TODO train a move encoder for text to data?
# TODO fix return type
def create_move_encoder(gen: str) -> dict:
    file_path = os.path.join(
        DATA_PATH,
        "moves_by_gen",
        f"{gen}_moves.json")
    with open(file_path, 'r') as f:
        data = json.load(f)

    moves = set()

    move_names = [name.lower() for name in move_names]
    move_names = [name.replace(" ", "") for name in move_names]

    print(f"move_count: {len(move_names)}")
    le = LabelEncoder()
    le.fit(move_names)
    return le


def encode_move(battle_format: str, move: str) -> int:
    if move is None or move == '':
        return 0
    return move_encoder.transform([move])[0] + 1

    if 'gen7' in battle_format:
        return GEN7_MOVE_ENCODER.transform([move])[0]
    else:
        raise ValueError("Invalid battle format")