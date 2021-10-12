import torch
from torch import nn
from icecream import ic
from typing import Tuple

from poke_env.environment.weather import Weather
from poke_env.environment.status import Status
from poke_env.environment.pokemon_type import PokemonType

from battler.utils.encoders.abilities import GEN8_POKEMON, GEN8_ABILITIES
from battler.utils.embed import MOVE_DIM, POKEMON_FLOAT_DIMS, POKEMON_ENUM_DIMS
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


from dataclasses import dataclass


# TODO move to passing in embeddings
@dataclass
class Embeddings:
    weather_embedding: nn.Embedding = nn.Embedding((len(Weather)+2), 2)
    name_embedding: nn.Embedding = nn.Embedding(len(GEN8_POKEMON)+2, 4)
    status_embedding: nn.Embedding = nn.Embedding(len(Status)+2, 2)
    type_embedding: nn.Embedding = nn.Embedding(len(PokemonType)+2, 4) # 1 higher for None
    #type2_embedding = _type1_embedding
    ability_embedding: nn.Embedding = nn.Embedding(272, 6)
    # TODO verify dims
    item_embedding: nn.Embedding = nn.Embedding(923, 8)

POKE_TEAM_OFFSET = (POKEMON_FLOAT_DIMS + POKEMON_ENUM_DIMS) * 6
MOVES_RANGE = MOVE_DIM * 4 * 6
ally_poke_end_idx = 1 + POKE_TEAM_OFFSET
opp_poke_end_idx = ally_poke_end_idx + POKE_TEAM_OFFSET
ally_moves_end_idx = opp_poke_end_idx + MOVES_RANGE
opp_moves_end_idx = ally_moves_end_idx + MOVES_RANGE

#class PokeMLP(nn.Module):

class PokesNet(nn.Module):
    def __init__(
        self,
        name_embedding,
        status_embedding,
        type_embedding,
        ability_embedding,
        item_embedding,
        per_poke_dim: int = 43,
        per_poke_out_dim = 8,
        per_poke_hidden_dim = 64,
        #in_dim: int = 49,
        hidden_dim = 64,
        out_dim = 8,

    ):
        super().__init__()
        self.name_embedding = name_embedding
        self.status_embedding = status_embedding
        self.type_embedding = type_embedding
        self.ability_embedding = ability_embedding
        self.item_embedding = item_embedding

        '''
        self.poke_net = nn.Sequential(
            nn.Linear(per_poke_dim, per_poke_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(per_poke_hidden_dim, per_poke_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(per_poke_hidden_dim, per_poke_out_dim),
            #nn.ReLU(),
        )
        self.team_net = nn.Sequential(
            nn.Linear(6*per_poke_out_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        '''


    def forward(self, x):
        x_ints = x[:, :, :36].long()

        names = self.name_embedding(x_ints[:, :, 0:6]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        status = self.status_embedding(x_ints[:, :, 6:12]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        type1 = self.type_embedding(x_ints[:, :, 12:18]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        type2 = self.type_embedding(x_ints[:, :, 18:24]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        ability = self.ability_embedding(x_ints[:, :, 24:30]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        item = self.item_embedding(x_ints[:, :, 30:36]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)

        pokes = x[:, :, 36:126].view(x_ints.shape[0], x_ints.shape[1], 6, 15)

        poke_x = torch.cat([pokes, names, status, type1, type2, ability, item], dim=-1)

        #poke_x = self.poke_net(poke_x)
        poke_x = poke_x.view(x_ints.shape[0], x_ints.shape[1], -1)
        #team_x = self.team_net(poke_x)
        #return team_x
        return poke_x


class MovesNet(nn.Module):
    def __init__(
        self,
        type_embedding,
        in_dim: int = 18,
        hidden_dim = 32,
        out_dim = 4,
    ):
        super().__init__()
        self.type_embedding = type_embedding
        '''
        self.move_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        '''

    def forward(self, x):
        moves_raw = x[:, : , :MOVES_RANGE].view(*x.shape[0:2], 6, 4, MOVE_DIM)
        int_in = moves_raw[:, :, :, :, 0].long()
        type_x = self.type_embedding(int_in)

        move_x = torch.cat((moves_raw[:, :, :, :, 1:], type_x), dim=-1)

        #x = self.move_net(x)
        move_x = move_x.view(moves_raw.shape[0], moves_raw.shape[1], -1)
        return move_x

class PokeMLP(BaseFeaturesExtractor):
    #NUM_TYPES = ##6
    '''
    layout:
    0: weather
    1-6: pokemons_names
    7-12: pokemons status
    13-18: pokemons types 1
    19-24: pokemons types 2
    25-30: pokemons abilities
    31-36: pokemons items
    31-36: pokemons items

    '''

    # TODO use layer input size here
        #10 + 2 + 3 + 3 + 6 + 10
    #ic(per_poke_enum_dim)

    # Net for pokemon info
    #ic(per_poke_embedding_dim)

    #_opp_per_poke_net = deepcopy(_per_poke_net)
    # TODO test opp_poke_embedding_net = deepcopy(poke_embedding_net)

    def __init__(
        self, 
        observation_space: Tuple[int], 
        n_actions: int, 
        weather_embedding: nn.Embedding = None,
        pokes_net: nn.Module = None,
        #opp_per_poke_net: nn.Module = None,
        moves_net: nn.Module = None,

        hidden_dim: int = 128,
        num_dense_layers: int = 0,
        conv_dim: int = 8,
        final_conv_dim: int = 2,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, n_actions)

        input_shape = observation_space.shape
        ic(observation_space)
        ic(input_shape)
        # I'm not sure why +2. +1 is to account for None
        self.weather_embedding = weather_embedding

        self.ally_pokes_net = pokes_net
        self.opp_pokes_net = pokes_net
        ic(self.ally_pokes_net)

        self.ally_moves_net = moves_net

        self.opp_moves_net = self.ally_moves_net
        ic(self.ally_moves_net)

        # TODO param poke_net

        # TODO keep testing reflex agents
        # TODO the agent does seem to behave differently based on previous attacks
        # TODO active poke net and other poke net

        # NOT exactly battle dims, it has encodings added
        # TODO why does torch work with lower dims on linear????

        # TODO  dynamically figure out frame_dim
        #frame_dim = input_shape[-1] + embedding_dim_delta + moves_dim_delta
        #frame_dim = 1530
        frame_dim = 1456

        layers = []

        # Stack depth expander
        #layers.append(nn.Conv1d(input_shape[0], conv_dim, 1))
        #layers.append(nn.LeakyReLU())

        # Frame dim reducer
        #layers.append(nn.Linear(frame_dim, frame_dim//4)) # filter down to half per frame
        #layers.append(nn.LeakyReLU())

        # TODO figure out conv looking set of 3 frames together for differences

        #layers.append(nn.Conv1d(conv_dim, final_conv_dim, 1))
        #layers.append(nn.LeakyReLU())
        layers.append(nn.Flatten(1))
        #layers.append(nn.Linear((frame_dim)*final_conv_dim, hidden_dim))
        #layers.append(nn.Linear((frame_dim//4)*final_conv_dim, hidden_dim))
        layers.append(nn.Linear((frame_dim)*input_shape[0], hidden_dim))
        layers.append(nn.LeakyReLU())
        for _ in range(num_dense_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            #layers.append(nn.LazyLinear(hidden_dim))
            layers.append(nn.LeakyReLU())
        #layers.append(nn.LazyLinear(hidden_dim, n_actions))
        #layers.append(nn.Linear(hidden_dim, n_actions))
        layers.append(nn.Linear(hidden_dim, n_actions))
        self.net = nn.Sequential(*layers)


        # Shuffle inforation inside frame
        #nn.Linear(frame_dim, frame_dim//2),
        #nn.Linear(hidden_size, hidden_size),
        #nn.ReLU(),

        # TODO make sure using eval mode other places if using batchnorm? not used elsewhere
        #ic(self.net)
    
    def forward(self, x):
        #if len(x.shape) == 4:
            # Needed because it one time gets 512x1x.x.
            #x.squeeze_(1)
        if len(x.shape) < 3:
            x.unsqueeze_(0)
        
        #x.squeeze_(0)
        #ic(x.shape)
        #x = x.swapaxes(1, -1)

        x_ints = torch.round(x[:, :, :1]).long()

        weather = self.weather_embedding(x_ints[:, :, 0:1]).view(x_ints.shape[0], x_ints.shape[1], -1)

        # Friendly Pokes
        poke_x = self.ally_pokes_net(x[:, :, 1:ally_poke_end_idx])
        # Opponent Pokes
        opp_poke_x = self.opp_pokes_net(x[:, :, ally_poke_end_idx:opp_poke_end_idx])

        # TODO handle type encoding in moves
        ally_move_x = self.ally_moves_net(x[:, :, opp_poke_end_idx:ally_moves_end_idx])

        opp_move_x = self.opp_moves_net(x[:, :, ally_moves_end_idx:opp_moves_end_idx])

        x = torch.cat((poke_x, opp_poke_x, weather, ally_move_x, opp_move_x, x[:, :, opp_moves_end_idx:]), dim=-1)

        #ic(x.shape)

        x = self.net(x)
        
        #x.squeeze_(0)
        #ic(x.shape)
        return x
 