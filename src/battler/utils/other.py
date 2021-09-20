from battler.players.opponents import RandomOpponentPlayer
from battler.players.opponents import RLOpponentPlayer
from battler.players.opponents import MaxDamagePlayer


def get_opponent(opponent_type: str):
    if opponent_type == 'random':
        return RandomOpponentPlayer
    elif opponent_type == 'self':
        return RLOpponentPlayer
    elif opponent_type == 'max_damage':
        return MaxDamagePlayer
    else:
        raise ValueError('Unknown opponent type: {}'.format(opponent_type))