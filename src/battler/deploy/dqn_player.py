import asyncio

from poke_env.player.random_player import RandomPlayer
from poke_env.player.player import Player
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration

from battler.deploy.showdown import main
from battler.players.opponents import build_eval_stocastic_policy
import torch.nn.functional as F


class DQNPlayer(Player):
    def __init__(self, *args, net, **kwargs):
        super().__init__(*args, **kwargs)
        net.eval()
        self.policy = build_eval_stocastic_policy(net)

    def choose_move(self, battle):
        action = self.policy(battle)
        return self._action_to_move(action, battle)

    def _action_to_move(self, action, battle):
        if (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action])
        elif (
            not battle.force_switch
            and battle.can_z_move
            and battle.active_pokemon
            and 0
            <= action - 4
            < len(battle.active_pokemon.available_z_moves)  # pyre-ignore
        ):
            return self.create_order(
                battle.active_pokemon.available_z_moves[action - 4], z_move=True
            )
        elif (
            battle.can_mega_evolve
            and 0 <= action - 8 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action - 8], mega=True)
        elif 0 <= action - 12 < len(battle.available_switches):
            return self.create_order(battle.available_switches[action - 12])
        else:
            return self.choose_random_move(battle)
if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main(DQNPlayer))