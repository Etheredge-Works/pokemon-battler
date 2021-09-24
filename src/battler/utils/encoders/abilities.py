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
    print(ability_names)
    le.fit(list(ability_names))
    return le
    

GEN7_ABILITY_ENCODER = create_ability_encoder("gen7")
GEN8_ABILITY_ENCODER = create_ability_encoder("gen8")
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
def create_pokemon_encoder(gen: str) -> dict:
    file_path = os.path.join(
        "data",
        "pokedex_by_gen",
        f"{gen}_pokedex.json")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    pokemon_names = set()
    for pokemon_name, pokemon_values in data.items():
        pokemon_name = pokemon_name.lower()\
            .replace(" ", "")\
            .replace("'", "")\
            .replace("-", "")\
            .replace("(", "")\
            .replace(")", "")
        pokemon_names.add(pokemon_name)
    

    le = LabelEncoder()
    le.fit(pokemon_names)
    return le


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
    le = LabelEncoder()
    le.fit(item_names)
    return le

GEN7_ITEM_ENCODER = create_item_encoder("gen7")
GEN7_ITEM_ENCODER = create_item_encoder("gen8")

def encode_item(battle_format: str, item: str) -> int:
    if item is None or item == '':
        return 0 
    return GEN7_ITEM_ENCODER.transform([item])[0] + 1

    if 'gen7' in battle_format:
        return GEN7_ITEM_ENCODER.transform([item])[0]
    else:
        raise ValueError("Invalid battle format")
