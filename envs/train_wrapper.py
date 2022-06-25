from ast import Raise
import nmmo
import numpy as np
from gym import Wrapper, spaces
from ijcai2022nmmo import TeamBasedEnv
from ijcai2022nmmo.scripted import CombatTeam, ForageTeam, RandomTeam
from ijcai2022nmmo.scripted.baselines import Scripted
from ijcai2022nmmo.scripted.scripted_team import ScriptedTeam
from copy import deepcopy
from envs.feature import FeatureParser
from envs.reward import RewardParser
from envs.move import MovePolicy
 
class TrainWrapper(Wrapper):
    max_step = 1024
    TT_ID = 0  # training team index
    use_auxiliary_script = False
    use_pretrained_move_net = True

    def __init__(self, env: TeamBasedEnv, team_sprit=0, action_type='move') -> None:
        super().__init__(env)
        self.feature_parser = FeatureParser()
        move_action_space = spaces.Discrete(5)
        attack_action_space = spaces.Discrete(61)
        self.action_type = action_type
        if action_type == 'move':
            self.action_space = move_action_space
        elif action_type == 'attack':
            self.action_space = attack_action_space
        elif action_type == 'both':
            self.action_space = spaces.MultiDiscrete([5, 61])
        self.agent_num = 8 # 控制的智能体数量
        self.reward_parser = RewardParser(team_sprit)
        if self.use_pretrained_move_net:
            self.move_policy = MovePolicy()
            from pathlib import Path
            import os 
            pth = Path(os.path.dirname(__file__)) / 'load_model' / 'actor.pt'
            self.move_policy.net.load(pth)
            
    def reset(self, random_team_id=False):
        if random_team_id:
            self.TT_ID = np.random.randint(0,16)
        raw_obs = super().reset()
        self.stage = np.zeros(8)
        raw_team_obs = raw_obs[self.TT_ID]
        obs, va, attack_index = self.feature_parser.parse(raw_team_obs)
        self.agents = list(raw_team_obs.keys())

        self.reset_auxiliary_script(self.config)
        self.reset_scripted_team(self.config)
        self._prev_achv = self.metrices_by_team()[self.TT_ID]
        self.reward_parser.reset(self._prev_achv)
        self._prev_raw_obs = raw_obs
        self._step = 0
        move_va = np.zeros((8,5))
        attack_va = np.zeros((8, 61))
        for agent_id in raw_team_obs:
           move_va[agent_id] = va['move_va'][agent_id]
           attack_va[agent_id] = va['attack_va'][agent_id]
        if self.action_type == 'move':
            available_actions = move_va
        elif self.action_type == 'attack':
            available_actions = attack_va
        elif self.action_space == 'both':
            available_actions = [move_va, attack_va]
        obs['global_obs'].update(time=np.array([0]))
        self.current_attack_index = attack_index
        info = self.reward_parser.max_prev_achv
        return obs, available_actions, info 

    def step(self, actions):
        
        # 得到其他队伍的决策
        decisions = self.get_scripted_team_decision(self._prev_raw_obs)
        # 得到自己队伍的决策
        decisions[self.TT_ID] = self.transform_action(
            self.action_type,
            actions,
            self.current_attack_index,
            observations=self._prev_raw_obs[self.TT_ID],
            auxiliary_script=self.auxiliary_script, 
            move_policy=self.move_policy
            )
        # 与环境交互
        raw_obs, _, raw_done, raw_info = super().step(decisions)
        self._step += 1
        # if agent die, will not return its obs
        if self.TT_ID in raw_obs:
            team_obs = raw_obs[self.TT_ID]
            done = raw_done[self.TT_ID]
            info = raw_info[self.TT_ID]
            obs, va, self.current_attack_index = self.feature_parser.parse(team_obs)
            obs['global_obs'].update(time=np.array([self._step/1024]))
            # compute reward
            achv = self.metrices_by_team()[self.TT_ID]
            reward = self.reward_parser.parse(achv=achv)
            self._prev_achv = achv
        else: 
            team_obs = {}
            done = {}
            reward = {}
            info = {}

        # 计算可行动作
        move_va = np.zeros((8,5))
        attack_va = np.zeros((8, 61))
        for agent_id in team_obs:
           move_va[agent_id] = va['move_va'][agent_id]
           attack_va[agent_id] = va['attack_va'][agent_id]
        if self.action_type == 'move':
            available_actions = move_va
        elif self.action_type == 'attack':
            available_actions = attack_va
        elif self.action_space == 'both':
            available_actions = [move_va, attack_va]
        self._prev_raw_obs = raw_obs
        
        if self._step >= self.max_step:
            done = {key: True for key in done.keys()}

        # 转成array数组
        reward_array = np.zeros((self.agent_num, 1))
        done_array = np.zeros((self.agent_num), dtype=bool)

        for agent_id in team_obs:
            done_array[agent_id] = done[agent_id]
            reward_array[agent_id] = reward[agent_id]
        
        for agent_id in self.agents:
            if agent_id not in team_obs:
                done_array[agent_id] = True 
                reward_array[agent_id] = 0

        info = self.reward_parser.max_prev_achv
        # 重置环境
        if np.all(done_array):
            obs, available_actions, _ = self.reset()
        return obs, reward_array, done_array, info, available_actions 

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
        # if getattr(self, "_scripted_team", None) is not None:
        #     for team in self._scripted_team.values():
        #         team.reset()
        #     return
        self._scripted_team = {}
        assert config.NPOP == 16
        enermy_team_id = np.delete(np.arange(config.NPOP), self.TT_ID)
        for index, id in enumerate(enermy_team_id):
            if index < 7:
                self._scripted_team[id] = RandomTeam(f"random-{id}", config)
            elif 7<= index < 12:
                self._scripted_team[id] = ForageTeam(f"forage-{id}", config)
            elif 12<= index < 15:
                self._scripted_team[id] = CombatTeam(f"combat-{id}", config)

    def get_scripted_team_decision(self, observations):
        decisions = {}
        tt_id = self.TT_ID
        for team_id, obs in observations.items():
            if team_id == tt_id:
                continue
            
            decisions[team_id] = self._scripted_team[team_id].act(obs)
        return decisions

    @staticmethod
    def transform_action(action_type, actions, all_target_index, observations=None, auxiliary_script=None, move_policy=None):
        """neural network move + scripted attack"""
        decisions = {}
        move_action = None
        raw_attack_action = None
        if action_type == 'move':
            move_action = actions 
        elif action_type == 'attack':
            raw_attack_action = actions
        elif action_type == 'both':
            move_action = actions[:,0]
            raw_attack_action = actions[:,1]
        # 预训练的move_policy
        if move_policy is not None:
            move_action = move_policy.get_decision(observations)
        # move decisions
        if move_action is not None:
            for agent_id, val in enumerate(move_action):
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
        if raw_attack_action is not None:
            for agent_id, val in enumerate(raw_attack_action):
                if observations is not None and agent_id not in observations:
                    continue
                if val != 0:
                    if val <= 20:
                        attack_action = {
                            nmmo.action.Style: 0,
                            nmmo.action.Target: int(all_target_index[agent_id][val - 1])
                        }
                    elif val <= 40:
                        attack_action = {
                            nmmo.action.Style: 1,
                            nmmo.action.Target: int(all_target_index[agent_id][val - 21])
                        }
                    elif val <= 60:
                        attack_action = {
                            nmmo.action.Style: 2,
                            nmmo.action.Target: int(all_target_index[agent_id][val - 41])
                        }
                    else:
                        raise ValueError(f"invalid attack action: {val}")
                    assert attack_action[nmmo.action.Target] != 0, 'check attack action transform !'
                    decisions[agent_id].update(                        
                        {
                                nmmo.action.Attack: attack_action
                            })
        # script attack decisions
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
