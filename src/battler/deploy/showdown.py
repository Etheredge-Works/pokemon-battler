import asyncio
import os

from poke_env.player.player import Player
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration

N_BATTLES = 100
MAX_CONCURRENT_BATTLE = 100


async def main(player_constructor: Player):
    # We create a random player
    player = player_constructor(
        player_configuration=PlayerConfiguration(os.environ["USERNAME"], os.environ["PASSWORD"]),
        server_configuration=ShowdownServerConfiguration,
        max_concurrent_battle=MAX_CONCURRENT_BATTLE,
    )

    await player.accept_challenges(None, N_BATTLES)
    for battle in player.battles.values():
        try:
            print(battle.rating, battle.opponent_rating)
        except:
            print(battle)
            pass