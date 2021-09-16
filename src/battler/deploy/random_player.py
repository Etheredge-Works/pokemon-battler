from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration
import os
import asyncio

async def main():
    # We create a random player
    player = RandomPlayer(
        player_configuration=PlayerConfiguration(os.environ["USERNAME2"], os.environ["PASSWORD"]),
        server_configuration=ShowdownServerConfiguration,
    )
    await player.accept_challenges(None, 5)
    for battle in player.battles.values():
        try:
            print(battle.rating, battle.opponent_rating)
        except:
            print(battle)
            pass

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())