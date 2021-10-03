import torch
from torch import nn
from icecream import ic
from typing import Tuple

from poke_env.environment.weather import Weather
from poke_env.environment.status import Status
from poke_env.environment.pokemon_type import PokemonType

from battler.utils.encoders.abilities import GEN8_POKEMON, GEN8_ABILITIES
from battler.utils.embed import MOVE_DIM, POKEMON_FLOAT_DIMS, POKEMON_ENUM_DIMS

class PokemonNet(nn.Module):
    def __init__(self,
        name_embedding_dim: int = 10,
        status_embedding_dim: int = 2,
        type_embedding_dim: int = 3,
        ability_embedding_dim: int = 6,
        item_embedding_dim: int = 10,
    ):
        super().__init__()
        self.name_embedding = nn.Embedding(len(GEN8_POKEMON)+2, 10)
        self.status_embedding = nn.Embedding(len(Status)+2, 2)

        # used twice
        self.type1_embedding = nn.Embedding(len(PokemonType)+2, 3) # 1 higher for None
        self.type2_embedding = self.type1_embedding
        #_type2_embedding = nn.Embedding(len(PokemonType)+2, 3) # 1 higher for None

        self.ability_embedding = nn.Embedding(272, 6)
        # TODO verify dims
        self.item_embedding = nn.Embedding(923, 10)

        self.per_poke_enum_dim = \
            self.name_embedding.embedding_dim + \
            self.type1_embedding.embedding_dim + \
            self.type2_embedding.embedding_dim + \
            self.ability_embedding.embedding_dim + \
            self.item_embedding.embedding_dim + \
            self.status_embedding.embedding_dim

        self.net = nn.Sequential(
            #nn.Conv1d(poke_embedding_dim, hidden_size, 1), # TODO convert to conv
            nn.Linear(self.per_poke_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
    
    def input_dim(self):
        return 126

    def output_dim(self):
        return 64

    def forward(self, x):
        x_ints = x[:, :, :36].long()

        names = self.name_embedding(x_ints[:, :, 0:6]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        status = self.status_embedding(x_ints[:, :, 6:12]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        type1 = self.type1_embedding(x_ints[:, :, 12:18]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        type2 = self.type2_embedding(x_ints[:, :, 18:24]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        ability = self.ability_embedding(x_ints[:, :, 24:30]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        item = self.item_embedding(x_ints[:, :, 30:36]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)

        pokes = x[:, :, 36:126].view(x_ints.shape[0], x_ints.shape[1], 6, 15)

        poke_x = torch.cat([pokes, names, status, type1, type2, ability, item], dim=-1)

        poke_x = self.per_poke_net(poke_x)
        poke_x = poke_x.view(x_ints.shape[0], x_ints.shape[1], -1)
        return poke_x


from dataclasses import dataclass


# TODO move to passing in embeddings
@dataclass
class Embeddings:
    weather_count = 1
    weather_embedding = nn.Embedding((len(Weather)+2), 2)
    name_count = 12
    name_embedding = nn.Embedding(len(GEN8_POKEMON)+2, 10)
    status_count = 12
    status_embedding = nn.Embedding(len(Status)+2, 2)
    # used twice
    #type_counts = 12 + (12 * 4 * ``)
    type_embedding = nn.Embedding(len(PokemonType)+2, 3) # 1 higher for None
    #_type2_embedding = nn.Embedding(len(PokemonType)+2, 3) # 1 higher for None
    ability_embedding = nn.Embedding(272, 6)
    # TODO verify dims
    item_embedding = nn.Embedding(923, 10)


POKE_TEAM_OFFSET = (POKEMON_FLOAT_DIMS + POKEMON_ENUM_DIMS) * 6
MOVES_RANGE = MOVE_DIM * 4 * 6
ally_poke_end_idx = 1 + POKE_TEAM_OFFSET
opp_poke_end_idx = ally_poke_end_idx + POKE_TEAM_OFFSET
ally_moves_end_idx = opp_poke_end_idx + MOVES_RANGE
opp_moves_end_idx = ally_moves_end_idx + MOVES_RANGE

class PokeMLP(nn.Module):
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
    _weather_embedding = nn.Embedding((len(Weather)+2), 4)
    ic(_weather_embedding)
    #ic(sum(_weather_embedding.parameters()))
    _name_embedding = nn.Embedding(len(GEN8_POKEMON)+2, 16)
    #ic(_name_embedding.parameters().numel())
    _status_embedding = nn.Embedding(len(Status)+2, 8)
    #ic(_status_embedding.parameters().numel())
    # used twice
    _type1_embedding = nn.Embedding(len(PokemonType)+2, 12) # 1 higher for None
    _type2_embedding = _type1_embedding
    #_type2_embedding = nn.Embedding(len(PokemonType)+2, 3) # 1 higher for None
    _ability_embedding = nn.Embedding(272, 16)
    # TODO verify dims
    _item_embedding = nn.Embedding(923, 16)

    # TODO use layer input size here
    per_poke_enum_dim = \
        _name_embedding.embedding_dim + \
        _type1_embedding.embedding_dim + \
        _type2_embedding.embedding_dim + \
        _ability_embedding.embedding_dim + \
        _item_embedding.embedding_dim + \
        _status_embedding.embedding_dim
    ic(per_poke_enum_dim)
        #10 + 2 + 3 + 3 + 6 + 10
    #ic(per_poke_enum_dim)

    # Net for pokemon info
    per_poke_embedding_dim = POKEMON_ENUM_DIMS + POKEMON_FLOAT_DIMS
    #ic(per_poke_embedding_dim)
    '''
    _per_poke_net = nn.Sequential(
        #nn.Conv1d(poke_embedding_dim, hidden_size, 1), # TODO convert to conv
        #nn.LazyLinear(per_poke_embedding_dim, 256),
        nn.Linear(per_poke_embedding_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        #nn.ReLU(),
    )

    #_opp_per_poke_net = deepcopy(_per_poke_net)
    _opp_per_poke_net = _per_poke_net
    # TODO test opp_poke_embedding_net = deepcopy(poke_embedding_net)

    _move_net = nn.Sequential(
        nn.Linear((move_dim-1) + _type1_embedding.embedding_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
    )
    '''



    def __init__(
        self, 
        input_shape: Tuple[int], 
        n_actions: int, 
        hidden_dim: int = 1024,
        num_dense_layers: int = 4,
        conv_dim: int = 32,
        final_conv_dim: int = 4,
    ):
        super().__init__()
        ic(input_shape)
        # I'm not sure why +2. +1 is to account for None
        self.weather_embedding = self._weather_embedding
        self.name_embedding = self._name_embedding
        self.status_embedding = self._status_embedding
        self.type1_embedding = self._type1_embedding
        self.type2_embedding = self._type2_embedding
        self.ability_embedding = self._ability_embedding
        self.item_embedding = self._item_embedding
        '''
        self.per_poke_net = nn.Sequential(
            #nn.Conv1d(poke_embedding_dim, hidden_size, 1), # TODO convert to conv
            #nn.LazyLinear(256),
            nn.Linear(self.per_poke_embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            #nn.ReLU(),
        )
        
        self.opp_poke_net = deepcopy(self.per_poke_net)
        ic(self.per_poke_net)
        #@self.opp_poke_net = deepcopy(self._opp_per_poke_net)
        #self.opp_poke_net = self.per_poke_net


        #self.move_net = deepcopy(self._move_net)
        self.move_net = nn.Sequential(
            nn.Linear((self.move_dim-1) + self._type1_embedding.embedding_dim, 64),
            #nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        #self.opp_move_net = deepcopy(self._move_net)
        self.opp_move_net = deepcopy(self.move_net)
        ic(self.move_net)

        # TODO param poke_net
        pre_converted_dims = 1 + ((15+6) * 12)
        ic(pre_converted_dims)
        #converted_dims = (self.per_poke_enum_dim * 12) + self.weather_embedding.embedding_dim
        post_converted_dims = (12 * 64) + self.weather_embedding.embedding_dim
        ic(post_converted_dims)

        #poke_dims = self.per_poke_net.
        embedding_dim_delta =  post_converted_dims - pre_converted_dims
        ic(embedding_dim_delta)
        conv_out_channels = 4

        #ic(input_shape)
        moves_flattened_dim = 6 * 4 * 8 * 2
        moves_dim_delta = moves_flattened_dim - (self.move_dim*12*4) - 1
        frame_dim = input_shape[1] + embedding_dim_delta + moves_dim_delta
        frame_dim = 2515
        net_input_dim = (input_shape[1] + embedding_dim_delta) * conv_out_channels
        net_input_dim = 2515 * 2
        ic(net_input_dim) # shoudl be 576
        '''
        # TODO keep testing reflex agents
        # TODO the agent does seem to behave differently based on previous attacks
        # TODO active poke net and other poke net

        total_dim = 1
        for dim in input_shape[0:]:
            total_dim *= dim
        
        # NOT exactly battle dims, it has encodings added
        # TODO why does torch work with lower dims on linear????
        total_dim = 2466
        frame_dim = total_dim

        layers = []

        # Stack depth expander
        layers.append(nn.Conv1d(input_shape[0], conv_dim, 1))
        layers.append(nn.LeakyReLU())

        # Frame dim reducer
        layers.append(nn.Linear(total_dim, total_dim//8)) # filter down to half per frame
        layers.append(nn.LeakyReLU())

        # TODO figure out conv looking set of 3 frames together for differences
        #layers.append(nn.Conv2d(conv_dim, conv_dim, (2, frame_dim)))
        #layers.append(nn.LeakyReLU())

        layers.append(nn.Conv1d(conv_dim, final_conv_dim, 1))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Flatten(1))
        layers.append(nn.Linear((total_dim//8)*final_conv_dim, hidden_dim))
        layers.append(nn.LeakyReLU())
        for _ in range(num_dense_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            #layers.append(nn.LazyLinear(hidden_dim))
            layers.append(nn.LeakyReLU())
        #layers.append(nn.LazyLinear(hidden_dim, n_actions))
        layers.append(nn.Linear(hidden_dim, n_actions))
        self.net = nn.Sequential(*layers)


        #self.net = nn.Sequential(
            #nn.Conv1d(input_shape[0], 64, kernel_size=1),
            #nn.BatchNorm1d(32),
            #nn.LazyLinear(hidden_dim),
            #nn.LeakyReLU(),
            #nn.LazyLinear(hidden_dim),
            #nn.LeakyReLU(),
            #nn.LazyLinear(hidden_dim),
            #nn.LeakyReLU(),
            #nn.LazyLinear(hidden_dim),
            #nn.LeakyReLU(),
            #nn.LazyLinear(hidden_dim),
            #nn.LeakyReLU(),
            # Shuffle inforation across frame
            #nn.Conv1d(input_shape[0], 64, kernel_size=1),
            #nn.BatchNorm1d(32),
            #nn.ReLU(),
            #nn.ReLU(),
            #nn.Conv1d(64, 64, kernel_size=1),
            #nn.BatchNorm1d(64),
            #nn.ReLU(),
            #nn.Linear(3103, 512),
            #nn.LazyLinear(1024),
            #nn.Conv1d(64, 64, kernel_size=1),
            #nn.BatchNorm1d(64),
            #nn.ReLU(),
            #nn.Linear(512, 256),
            #nn.Linear(frame_dim, frame_dim//2),
            #nn.LazyLinear(512),
            #nn.ReLU(),

            # Shuffle inforation inside frame
            #nn.Linear(frame_dim, frame_dim//2),
            #nn.Linear(hidden_size, hidden_size),
            #nn.ReLU(),

            #nn.Conv1d(64, 64, kernel_size=1),
            # TODO make sure using eval mode other places if using batchnorm? not used elsewhere
            #nn.BatchNorm1d(64),
            #nn.ReLU(),
            #nn.AdaptiveAvgPool1d(64),
            #nn.Flatten(),
            #nn.Linear(net_input_dim, hidden_size),
            #nn.Linear(hidden_size*conv_out_channels, hidden_size),
            #nn.Linear(hidden_size*conv_out_channels, hidden_size),
            #nn.LazyLinear(hidden_size),
            #nn.Dropout(0.2),
            #nn.Linear(net_input_dim//conv_out_channels, hidden_size),
            #nn.ReLU(),
            #nn.Linear(hidden_size, hidden_size),
            #nn.LazyLinear(hidden_size),
            #nn.ReLU(),
            #nn.LazyLinear(n_actions),
        #)
        ic(self.net)
        self.moves_range = MOVE_DIM * 4 * 6 

        self.per_poke_net = None
        self.opp_poke_net = None

        self.opp_move_net = None
        self.move_net = None
        self.opp_move_net = None

        #net_input_dim = (input_shape[0] * input_shape[1])
        #net_input_dim = 10


    def move_forward(self, x, net):
        moves_raw = x[:, : , :self.moves_range].view(*x.shape[0:2], 6, 4, MOVE_DIM)
        ##ic(moves_raw.shape)
        int_in = moves_raw[:, :, :, :, 0].long()
        #ic(int_in.shape)
        type_x = self.type1_embedding(int_in)
        #ic(type_x.shape)
        x = torch.cat((moves_raw[:, :, :, :, 1:], type_x), dim=-1)
        #ic(x.shape)
        return x
        y = net(x)
        return y
        #ic(y.shape)


    def poke_forward(self, x, net):
        x_ints = x[:, :, :36].long()

        names = self.name_embedding(x_ints[:, :, 0:6]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        status = self.status_embedding(x_ints[:, :, 6:12]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        type1 = self.type1_embedding(x_ints[:, :, 12:18]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        type2 = self.type2_embedding(x_ints[:, :, 18:24]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        ability = self.ability_embedding(x_ints[:, :, 24:30]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)
        item = self.item_embedding(x_ints[:, :, 30:36]).view(x_ints.shape[0], x_ints.shape[1], 6, -1)

        pokes = x[:, :, 36:126].view(x_ints.shape[0], x_ints.shape[1], 6, 15)

        poke_x = torch.cat([pokes, names, status, type1, type2, ability, item], dim=-1)

        #poke_x = net(poke_x)
        poke_x = poke_x.view(x_ints.shape[0], x_ints.shape[1], -1)
        return poke_x
    
    def forward(self, x):
        if len(x.shape) == 4:
            # Needed because it one time gets 512x1x.x.
            x.squeeze_(1)
        elif len(x.shape) < 3:
            x.unsqueeze_(0)

        x_ints = torch.round(x[:, :, :1]).long()

        weather = self.weather_embedding(x_ints[:, :, 0:1]).view(x_ints.shape[0], x_ints.shape[1], -1)

        # Friendly Pokes
        poke_x = self.poke_forward(x[:, :, 1:ally_poke_end_idx], self.per_poke_net)
        # Opponent Pokes
        opp_poke_x = self.poke_forward(x[:, :, ally_poke_end_idx:opp_poke_end_idx], self.opp_poke_net)

        # TODO handle type encoding in moves
        ally_move_x = self.move_forward(x[:, :, opp_poke_end_idx:ally_moves_end_idx], self.move_net)
        ally_move_x = ally_move_x.view(x_ints.shape[0], x_ints.shape[1], -1)

        opp_move_x = self.move_forward(x[:, :, ally_moves_end_idx:opp_moves_end_idx], self.opp_move_net)
        opp_move_x = opp_move_x.view(x_ints.shape[0], x_ints.shape[1], -1)

        x = torch.cat((poke_x, opp_poke_x, weather, ally_move_x, opp_move_x, x[:, :, opp_moves_end_idx:]), dim=-1)

        x = self.net(x)
        
        x.squeeze_(0)
        return x
 