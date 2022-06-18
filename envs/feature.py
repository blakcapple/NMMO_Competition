from gym import spaces
import numpy as np 

class FeatureParser:
    map_size = 15
    move_n_actions = 5 # 4 + 1
    attack_n_actions = 61 # 20*3 + 1
    NEIGHBOR = [(6, 7), (8, 7), (7, 8), (7, 6)]  # north, south, east, west
    OBSTACLE = (0, 1, 5)  # lava, water, stone
    COMBAT_MAGE_REACH = 4
    COMBAT_RANGE_REACH = 3
    COMBAT_MELEE_REACH = 1

    @staticmethod
    def _onehot_initialization(a, num_class=None):
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
        local_map = np.zeros((8, 17, 15, 15), dtype=np.float32) # 以agent为中心的空间信息
        agent_vector = np.zeros((8, 17), dtype=np.float32) # agent 本身的信息
        npc_vector = np.zeros((20, 17), dtype=np.float32) # team观察到的所有npc信息
        enemy_vector = np.zeros((20, 17), dtype=np.float32) # team观察到的所有的敌人的信息
        attack_vector = np.zeros((8, 20, 12), dtype=np.float32) # agent观察到的能够攻击的npc和敌人信息
        move_va_vector = np.zeros((8, self.move_n_actions), dtype=np.float32)
        attack_va_vector = np.zeros((8, self.attack_n_actions), dtype=np.float32)
        npc_id = []
        enemy_id = []
        npc_num = 0
        player_num = 0
        for agent_id in obs:
            terrain = np.zeros((self.map_size, self.map_size), dtype=np.int64)
            camp = np.zeros((self.map_size, self.map_size), dtype=np.int64)
            entity = np.zeros((7, self.map_size, self.map_size),
                              dtype=np.float32)
            local_attack = []

            move_va = np.ones(self.move_n_actions, dtype=np.int64)
            attack_va = np.ones(self.attack_n_actions, dtype=np.int64)

            tile = obs[agent_id]["Tile"]["Continuous"]
            LT_R, LT_C = tile[0, 2], tile[0][3] # left top index (absolute)
            for line in tile:
                terrain[int(line[2] - LT_R),
                        int(line[3] - LT_C)] = int(line[1])
            raw_entity = obs[agent_id]["Entity"]["Continuous"]
            P = raw_entity[0, 4]
            for index, line in enumerate(raw_entity):
                # local map feature 
                if line[0] != 1:
                    continue
                raw_pop, raw_r, raw_c = line[4:7]
                r, c = int(raw_r - LT_R), int(raw_c - LT_C) # 在智能体视野内的相对位置坐标
                camp[r, c] = 2 if raw_pop == P else np.sign(raw_pop) + 2 # none 0; npc 1; team 2; other 3 
                # level
                entity[0, r, c] = line[3] / 10 
                # damage, timealive, food, water, health, is_freezed
                entity[1, r, c] = line[7] / 10 
                entity[2, r, c] = line[8] / 1024 
                entity[3, r, c] = line[9] / 10 
                entity[4, r, c] = line[10] / 10
                entity[5, r, c] = line[11] / 10

                info = np.zeros(17, dtype=np.float32)
                info[0] = line[3] / 10.0 
                info[1] = line[5] / 128 
                info[2] = line[6] / 128
                info[3] = line[7] / 10
                info[4] = line[8] / 1024 
                info[5] = line[9] / 10 
                info[6] = line[10] / 10
                info[7] = line[11] / 10
                info[8] = line[12]
                info[9+agent_id] = 1 # 向量末尾加上agent one-hot编码
                if line[4] != P:
                    # 非队友信息
                    attack_info = np.zeros(12, dtype=np.float32)
                    attack_info[0] = 1 
                    if line[4] < 0:
                        attack_info[1] = 1
                    else:
                        attack_info[2] = 1
                    attack_info[3:] = info[:9]
                    local_attack.append(attack_info)
                if index == 0: 
                    # agent info 
                    agent_vector[agent_id] = info
                elif line[1] < 0 and line[1] not in npc_id:
                    # npc info
                    npc_id.append(line[1]) 
                    npc_num += 1
                    if npc_num > 20:
                        break
                    npc_vector[npc_num-1] = info
                elif line[1] > 0 and line[1] not in enemy_id and line[4] != P:
                    # enemy info
                    enemy_id.append(line[1])
                    player_num += 1
                    if player_num > 20:
                        break  
                    enemy_vector[player_num-1] = info
            
            # valid move_action
            for i, (r, c) in enumerate(self.NEIGHBOR):
                if terrain[r, c] in self.OBSTACLE:
                    move_va[i + 1] = 0
                    
            terrain = self._onehot_initialization(terrain, num_class=6).transpose(2,0,1)
            camp = self._onehot_initialization(camp, num_class=4).transpose(2,0,1)
            local_feature = np.concatenate([terrain, camp, entity], axis=0)
            local_map[agent_id] = local_feature
            for i, value in enumerate(local_attack):
                if i >= 20:
                    break
                attack_vector[agent_id][i] = value 
            # valid attack_action 
            # 注意攻击距离的判断，当处于同一个位置或者超过攻击范围时都不能打击;
            # 打击距离 = max(np.abs(start, end)) 取x和y轴距离的最大值
            agent_pos = agent_vector[agent_id][1:3]
            for i, value in enumerate(attack_vector[agent_id]):
                if value[0] != 1:
                    attack_va[i] = 0
                    attack_va[i + 20] = 0
                    attack_va[i + 40] = 0
                else:
                    target_pos = value[4:6]
                    # 智能体到目标的距离
                    distance = max(128 * np.abs(target_pos - agent_pos))
                    if distance > self.COMBAT_MAGE_REACH:
                        attack_va[i] = 0
                        attack_va[i + 20] = 0
                        attack_va[i + 40] = 0
                    elif distance > self.COMBAT_RANGE_REACH:
                        attack_va[i] = 0
                        attack_va[i + 20] = 0
                    elif distance > self.COMBAT_MELEE_REACH:
                        attack_va[i] = 0
                    else:
                        # 距离等于0，三种方式都不可以攻击
                        if distance == 0:
                            attack_va[i] = 0
                            attack_va[i + 20] = 0
                            attack_va[i + 40] = 0
            move_va_vector[agent_id] = move_va
            attack_va_vector[agent_id] = attack_va
            
        return {'agent_vector': agent_vector,
                'team_vector': agent_vector.reshape(-1),
                'enermy_vector': enemy_vector.reshape(-1),
                'npc_vector': npc_vector.reshape(-1), 
                'attack_vector': attack_vector.reshape(8,-1),
                'local_map': local_map,
                'move_va': move_va_vector,
                'attack_va': attack_va_vector}












