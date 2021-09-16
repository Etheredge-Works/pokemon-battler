import asyncio

from poke_env.player.random_player import RandomPlayer
from poke_env.player.player import Player
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration

from battler.deploy.showdown import main


def actual_power(move):
    return move.base_power * move.accuracy


class BasePlayer(Player):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: actual_power(move))
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main(BasePlayer))