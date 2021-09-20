from poke_env.environment.pokemon import Pokemon


class MoveEmbed:
    pass
class PokemonEmbed:
    def __init__(self, pokemon):
        self.pokemon = pokemon
        self.embed = self.build_embed()
    pass