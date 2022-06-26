import torch.nn as nn 
from algorithms.utils.util import init
import torch 
import torch.nn.functional as F
import numpy as np 
from algorithms.utils.act import ACTLayer
from gym import spaces
import nmmo
from ijcai2022nmmo.scripted.baselines import Scripted
from ijcai2022nmmo.scripted.scripted_team import ScriptedTeam

class CNNLayer(nn.Module):
    def __init__(self, use_orthogonal=True, use_ReLU=True):
        super(CNNLayer, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=17,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1)), nn.ReLU(),
            init_(nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1)), nn.ReLU(),
            init_(nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1)), nn.ReLU(),
            init_(nn.Conv2d(in_channels=32,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1)), nn.ReLU(),
            init_(nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1)), nn.ReLU())
        self.core = (nn.Linear(16*15*15, 512))


    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.core(x))
        return x

class CNNBase(nn.Module):
    def __init__(self):
        super(CNNBase, self).__init__()

        self._use_orthogonal = True
        self._use_ReLU = True

        self.cnn = CNNLayer(self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        x = self.cnn(x)
        return x

class MoveNet(nn.Module):

    def __init__(self):

        super().__init__()
        self.base = CNNBase()
        self.act = ACTLayer(spaces.Discrete(5), 512, True, 0)

    def forward(self, x, va):
        feature = self.base(x)
        action = self.act(feature, va)
        return action 

    def load(self, pth):
    
        self.load_state_dict(torch.load(pth, map_location='cpu'))

class Attack(Scripted):
    '''attack'''
    name = 'Attack_'

    def __call__(self, obs):
        super().__call__(obs)

        self.scan_agents()
        self.target_weak()
        self.style = nmmo.action.Range
        self.attack()
        return self.actions


class AttackTeam(ScriptedTeam):
    agent_klass = Attack

class MovePolicy(): 

    net = MoveNet()
    map_size = 15
    n_actions = 5
    NEIGHBOR = [(6, 7), (8, 7), (7, 8), (7, 6)]  # north, south, east, west
    OBSTACLE = (0, 1, 5)  # lava, water, stone
    
    def __init__(self,config):

        self.auxiliary_script = AttackTeam("auxiliary", config)

    def _onehot_initialization(self, a, num_class=None):
        """
        把输入的numpy数组转成one-hot (没有完全看懂这个实现方式)
        """
        def _all_idx(idx, axis):
            grid = np.ogrid[tuple(map(slice, idx.shape))]
            grid.insert(axis, idx)
            return tuple(grid)
        if num_class == None:
            ncols = a.max() + 1
        else: 
            ncols = num_class
        out = np.zeros(a.shape + (ncols,), dtype=int)
        out[_all_idx(a, axis=2)] = 1 
        return out

    def parse(self, obs):
        ret = {}
        for agent_id in obs:
            terrain = np.zeros((self.map_size, self.map_size), dtype=np.int64)
            camp = np.zeros((self.map_size, self.map_size), dtype=np.int64)
            entity = np.zeros((7, self.map_size, self.map_size),
                              dtype=np.float32)
            va = np.ones(self.n_actions, dtype=np.int64)

            # terrian feature
            tile = obs[agent_id]["Tile"]["Continuous"]
            LT_R, LT_C = tile[0, 2], tile[0][3] # left top index (absolute)
            for line in tile:
                terrain[int(line[2] - LT_R),
                        int(line[3] - LT_C)] = int(line[1])

            # npc and player
            raw_entity = obs[agent_id]["Entity"]["Continuous"]
            P = raw_entity[0, 4]
            _r, _c = raw_entity[0, 5:7]  # agent itself is at the center of observation map [7,7]
            assert int(_r - LT_R) == int(
                _c - LT_C) == 7, f"({int(_r - LT_R)}, {int(_c - LT_C)})"
            for line in raw_entity:
                if line[0] != 1:
                    continue
                raw_pop, raw_r, raw_c = line[4:7]
                r, c = int(raw_r - LT_R), int(raw_c - LT_C) # 在智能体视野内的相对位置坐标
                camp[r, c] = 2 if raw_pop == P else np.sign(raw_pop) + 2 # none 0; npc 1; team 2; other 3 
                # level
                entity[0, r, c] = line[3]
                # damage, timealive, food, water, health, is_freezed
                entity[1:, r, c] = line[7:]

            # valid action
            for i, (r, c) in enumerate(self.NEIGHBOR):
                if terrain[r, c] in self.OBSTACLE:
                    va[i + 1] = 0

            ret[agent_id] = {
                "terrain": terrain,
                "camp": camp,
                "entity": entity,
                "va": va
            }
        return ret

    def obs_transform(self, obs):
        features = np.zeros((8, 17, 15, 15)) # 对于死亡的智能体，其特征向量全为零
        for agent_id in obs:
            terrain, camp, entity = obs[agent_id]["terrain"], obs[agent_id]["camp"], obs[agent_id]["entity"]
            # shape [n, 15*15]
            terrain = self._onehot_initialization(terrain, num_class=6).transpose(2,0,1)
            camp = self._onehot_initialization(camp, num_class=4).transpose(2,0,1)
            feature = np.concatenate([terrain, camp, entity], axis=0)
            features[agent_id] = feature
        return features

    def act(self, raw_obs):
        obs = self.parse(raw_obs)
        va = np.zeros((8, self.n_actions), dtype=np.int64)
        for id in obs.keys():
            va[id] = obs[id]['va']
        feature = self.obs_transform(obs)
        feature = torch.from_numpy(feature).to(dtype=torch.float32)
        va = torch.from_numpy(va).to(dtype=torch.float32)
        actions,_ = self.net(feature, va)
        decisions = {}
        for agent_id, val in enumerate(actions):
            if raw_obs is not None and agent_id not in raw_obs:
                continue
            if val == 0:
                decisions[agent_id] = {}
            elif 1 <= val <= 4:
                decisions[agent_id] = {
                    nmmo.action.Move: {
                        nmmo.action.Direction: int(val) - 1
                    }
                }
            else:
                raise ValueError(f"invalid action: {val}")
            attack_decisions = self.auxiliary_script.act(raw_obs)
            # merge decisions
            for agent_id, d in decisions.items():
                d.update(attack_decisions[agent_id])
                decisions[agent_id] = d
        return decisions 

    def reset(self):
        self.auxiliary_script.reset()