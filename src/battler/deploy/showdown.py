import asyncio
import os

from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration


async def main(player_constructor):
    # We create a random player
    player = player_constructor(
        player_configuration=PlayerConfiguration(os.environ["USERNAME"], os.environ["PASSWORD"]),
        server_configuration=ShowdownServerConfiguration,
    )
    await player.accept_challenges(None, 5)
    for battle in player.battles.values():
        try:
            print(battle.rating, battle.opponent_rating)
        except:
            print(battle)
            pass