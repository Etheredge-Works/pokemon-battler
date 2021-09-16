
import asyncio
import time

from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer


class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

def actual_power(move):
    return move.base_power * move.accuracy



class TestPlayer(Player):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: actual_power(move))
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


async def main():
    start = time.time()

    # We create two players.
    random_player = RandomPlayer(
        battle_format="gen8randombattle",
    )
    custom_player = TestPlayer(
        battle_format="gen8randombattle",
    )
    max_damage_player = MaxDamagePlayer(
        battle_format="gen8randombattle",
    )

    n_battles = 4000
    # Now, let's evaluate our player
    await custom_player.battle_against(max_damage_player, n_battles=n_battles)

    
    b_time = time.time() - start
    print(
        f"Custom damage player won {custom_player.n_won_battles} / {custom_player.n_finished_battles} battles [this took {b_time} seconds]"
    )


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
