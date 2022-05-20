from matplotlib.style import available
import nmmo
import numpy as np
from gym import Wrapper, spaces
from ijcai2022nmmo import TeamBasedEnv
from ijcai2022nmmo.scripted import CombatTeam, ForageTeam, RandomTeam
from ijcai2022nmmo.scripted.baselines import Scripted
from ijcai2022nmmo.scripted.scripted_team import ScriptedTeam
from copy import deepcopy

from wandb import agent


class FeatureParser:
    map_size = 15
    n_actions = 5
    NEIGHBOR = [(6, 7), (8, 7), (7, 8), (7, 6)]  # north, south, east, west
    OBSTACLE = (0, 1, 5)  # lava, water, stone
    feature_spec = {
        "terrain": spaces.Box(low=0, high=6, shape=(15, 15), dtype=np.int64),
        "camp": spaces.Box(low=0, high=4, shape=(15, 15), dtype=np.int64),
        "entity": spaces.Box(low=0,
                             high=4,
                             shape=(7, 15, 15),
                             dtype=np.float32),
        "va": spaces.Box(low=0, high=2, shape=(5, ), dtype=np.int64), # valid action
    }

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


class RewardParser:

    def parse(self, prev_achv, achv):
        reward = {
            i: (sum(achv[i].values()) - sum(prev_achv[i].values())) / 100.0
            for i in achv
        }
        return reward


class TrainWrapper(Wrapper):
    max_step = 1024
    TT_ID = 0  # training team index
    use_auxiliary_script = True

    def __init__(self, env: TeamBasedEnv) -> None:
        super().__init__(env)
        self.feature_parser = FeatureParser()
        self.reward_parser = RewardParser()
        self.observation_space = spaces.Box(low=0, high=1, shape=(17, 15, 15))
        self.share_observation_space = self.observation_space 
        self.action_space = spaces.Discrete(5)
        self.agent_num = 8 # 控制的智能体数量

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

    def obs_transform(self, obs):
        features = np.zeros((self.agent_num, 17, 15, 15)) # 对于死亡的智能体，其特征向量全为零
        for agent_id in obs:
            terrain, camp, entity = obs[agent_id]["terrain"], obs[agent_id]["camp"], obs[agent_id]["entity"]
            # shape [n, 15*15]
            terrain = self._onehot_initialization(terrain, num_class=6).transpose(2,0,1)
            camp = self._onehot_initialization(camp, num_class=4).transpose(2,0,1)
            feature = np.concatenate([terrain, camp, entity], axis=0)
            features[agent_id] = feature
        return features

    def reset(self):
        raw_obs = super().reset()
        obs = raw_obs[self.TT_ID]
        obs = self.feature_parser.parse(obs)
        self.agents = list(obs.keys())

        self.reset_auxiliary_script(self.config)
        self.reset_scripted_team(self.config)
        self._prev_achv = self.metrices_by_team()[self.TT_ID]
        self._prev_raw_obs = raw_obs
        self._step = 0
        obs_array = self.obs_transform(obs) # 输出shape：[num, 17, 15, 15]
        share_obs_array = deepcopy(obs_array)
        available_actions = np.zeros((8,5))
        for agent_id in obs:
           available_actions[agent_id] = obs[agent_id]['va']
        return obs_array, share_obs_array, available_actions

    def step(self, actions):
        
        # 得到其他队伍的决策
        decisions = self.get_scripted_team_decision(self._prev_raw_obs)
        # 得到自己队伍的决策
        decisions[self.TT_ID] = self.transform_action(
            actions,
            observations=self._prev_raw_obs[self.TT_ID],
            auxiliary_script=self.auxiliary_script)

        raw_obs, _, raw_done, raw_info = super().step(decisions)
        # if agent die, will not return its obs
        if self.TT_ID in raw_obs:
            obs = raw_obs[self.TT_ID]
            done = raw_done[self.TT_ID]
            info = raw_info[self.TT_ID]
            obs = self.feature_parser.parse(obs)
            # compute reward
            achv = self.metrices_by_team()[self.TT_ID]
            reward = self.reward_parser.parse(self._prev_achv, achv)
            self._prev_achv = achv
        else: 
            obs = {}
            done = {}
            reward = {}
            info = {}

        available_actions = np.zeros((8,5))
        for agent_id in obs:
           available_actions[agent_id] = obs[agent_id]['va']     
        self._prev_raw_obs = raw_obs
        self._step += 1

        if self._step >= self.max_step:
            done = {key: True for key in done.keys()}

        # 转成array数组
        obs_array = self.obs_transform(obs)
        share_obs_array = deepcopy(obs_array) # 这里引入share_obs, 只是为了适应mappo的代码框架
        reward_array = np.zeros((self.agent_num, 1))
        done_array = np.zeros((self.agent_num))

        for agent_id in obs:
            done_array[agent_id] = done[agent_id]
            reward_array[agent_id] = reward[agent_id]
        
        for agent_id in self.agents:
            if agent_id not in obs:
                done_array[agent_id] = True 
                reward_array[agent_id] = 0

        # note: info contain true evaluation metrix !!
        return obs_array, share_obs_array, reward_array, done_array, info, available_actions 

    def reset_auxiliary_script(self, config):
        """
        重置自动攻击脚本
        """
        if not self.use_auxiliary_script:
            self.auxiliary_script = None
            return
        if getattr(self, "auxiliary_script", None) is not None:
            self.auxiliary_script.reset()
            return
        self.auxiliary_script = AttackTeam("auxiliary", config)

    def reset_scripted_team(self, config):
        """
        重置脚本队伍
        """
        if getattr(self, "_scripted_team", None) is not None:
            for team in self._scripted_team.values():
                team.reset()
            return
        self._scripted_team = {}
        assert config.NPOP == 16
        for i in range(config.NPOP):
            if i == self.TT_ID:
                continue
            if self.TT_ID < i <= self.TT_ID + 7:
                self._scripted_team[i] = RandomTeam(f"random-{i}", config)
            elif self.TT_ID + 7 < i <= self.TT_ID + 12:
                self._scripted_team[i] = ForageTeam(f"forage-{i}", config)
            elif self.TT_ID + 12 < i <= self.TT_ID + 15:
                self._scripted_team[i] = CombatTeam(f"combat-{i}", config)

    def get_scripted_team_decision(self, observations):
        decisions = {}
        tt_id = self.TT_ID
        for team_id, obs in observations.items():
            if team_id == tt_id:
                continue
            decisions[team_id] = self._scripted_team[team_id].act(obs)
        return decisions

    @staticmethod
    def transform_action(actions, observations=None, auxiliary_script=None):
        """neural network move + scripted attack"""
        decisions = {}

        # move decisions
        for agent_id, val in enumerate(actions):
            if observations is not None and agent_id not in observations:
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

        # attack decisions
        if auxiliary_script is not None:
            assert observations is not None
            attack_decisions = auxiliary_script.act(observations)
            # merge decisions
            for agent_id, d in decisions.items():
                d.update(attack_decisions[agent_id])
                decisions[agent_id] = d
        return decisions

    def close(self):
        self.close()

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
