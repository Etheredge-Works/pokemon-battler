import asyncio

from poke_env.player.random_player import RandomPlayer
from battler.deploy.showdown import main


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main(RandomPlayer))