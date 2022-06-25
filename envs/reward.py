
class RewardParser:
    keys = ['TimeAlive', 'Exploration', 'Equipment', 'PlayerDefeats', 'Foraging']

    def __init__(self, team_spirit):

        self.team_spirit = team_spirit
    
    def reset(self, achv):
        self.prev_achv = achv 
        self.max_prev_achv = {key:0 for key in self.keys}
        for key in self.keys:
            self.max_prev_achv[key] = max([achv[i][key] for i in achv])
        self.last_team_stage = {key:0 for key in self.keys}
        self.last_agent_stage = {i:{key:0 for key in self.keys} for i in achv}

    def update_team_stage(self, achv):
        team_stage = {}
        if achv['Exploration'] >= 127:
            team_stage['Exploration'] = 3
        elif achv['Exploration'] >= 64:
            team_stage['Exploration'] = 2
        elif achv['Exploration'] >=32:
            team_stage['Exploration'] = 1
        else:
            team_stage['Exploration'] = 0

        if achv['Equipment'] >= 20:
            team_stage['Equipment'] = 3
        elif achv['Equipment'] >= 10:
            team_stage['Equipment'] = 2
        elif achv['Equipment'] >= 1:
            team_stage['Equipment'] = 1
        else:
            team_stage['Equipment'] = 0

        if achv['PlayerDefeats'] >= 6:
            team_stage['PlayerDefeats'] = 3
        elif achv['PlayerDefeats'] >= 3:
            team_stage['PlayerDefeats'] = 2
        elif achv['PlayerDefeats'] >= 1:
            team_stage['PlayerDefeats'] = 1 
        else:
            team_stage['PlayerDefeats'] = 0 

        if achv['Foraging'] >= 50:
            team_stage['Foraging'] = 3
        elif achv['Foraging'] >= 35:
            team_stage['Foraging'] = 2
        elif achv['Foraging'] >= 20:
            team_stage['Foraging'] = 1
        else:
            team_stage['Foraging'] = 0

        if achv['TimeAlive'] >= 1000:
            team_stage['TimeAlive'] = 3
        elif achv['TimeAlive'] >= 500:
            team_stage['TimeAlive'] = 2
        elif achv['TimeAlive'] >= 250:
            team_stage['TimeAlive'] = 1
        else:
            team_stage['TimeAlive'] = 0
        return team_stage

    def update_agent_stage(self, achv):
        all_stage = {}
        for i in achv:
            stage = {}
            if achv[i]['Exploration'] >= 127:
                stage['Exploration'] = 3
            elif achv[i]['Exploration'] >= 64:
                stage['Exploration'] = 2
            elif achv[i]['Exploration'] >=32:
                stage['Exploration'] = 1
            else:
                stage['Exploration'] = 0

            if achv[i]['Equipment'] >= 20:
                stage['Equipment'] = 3
            elif achv[i]['Equipment'] >= 10:
                stage['Equipment'] = 2
            elif achv[i]['Equipment'] >= 1:
                stage['Equipment'] = 1
            else:
                stage['Equipment'] = 0

            if achv[i]['PlayerDefeats'] >= 6:
                stage['PlayerDefeats'] = 3
            elif achv[i]['PlayerDefeats'] >= 3:
                stage['PlayerDefeats'] = 2
            elif achv[i]['PlayerDefeats'] >= 1:
                stage['PlayerDefeats'] = 1 
            else:
                stage['PlayerDefeats'] = 0 

            if achv[i]['Foraging'] >= 50:
                stage['Foraging'] = 3
            elif achv[i]['Foraging'] >= 35:
                stage['Foraging'] = 2
            elif achv[i]['Foraging'] >= 20:
                stage['Foraging'] = 1
            else:
                stage['Foraging'] = 0 

            if achv[i]['TimeAlive'] >= 1000:
                stage['TimeAlive'] = 3
            elif achv[i]['TimeAlive'] >= 500:
                stage['TimeAlive'] = 2
            elif achv[i]['TimeAlive'] >= 250:
                stage['TimeAlive'] = 1
            else:
                stage['TimeAlive'] = 0

            all_stage[i] = stage
        return all_stage

    def compute_team_reward(self, prev_achv, achv):
        reward = 0 
        alive_reward = (achv['TimeAlive'] - prev_achv['TimeAlive']) / 102.4
        explore_reward = (achv['Exploration'] - prev_achv['Exploration'])/ 12.7   
        equip_reward = (achv['Equipment'] - prev_achv['Equipment'])/ 2
        # defeat_reward = (achv['PlayerDefeats'] - prev_achv['PlayerDefeats'])/ 0.6
        forag_reward = (achv['Foraging'] - prev_achv['Foraging'])/ 5
        reward = alive_reward + explore_reward + equip_reward + forag_reward
        # for key in self.keys:
        #     reward += (self.team_stage[key] - self.last_team_stage[key])*10
        return reward 

    def compute_agent_reward(self, prev_achv, achv):

        reward = {i: 0 for i in achv}
        for i in achv:
            alive_reward = (achv[i]['TimeAlive'] - prev_achv[i]['TimeAlive']) / 102.4
            explore_reward = (achv[i]['Exploration'] - prev_achv[i]['Exploration'])/ 12.7   
            equip_reward = (achv[i]['Equipment'] - prev_achv[i]['Equipment'])/ 2
            # defeat_reward = (achv[i]['PlayerDefeats'] - prev_achv[i]['PlayerDefeats'])/ 0.6
            forag_reward = (achv[i]['Foraging'] - prev_achv[i]['Foraging'])/ 5
            reward[i] = alive_reward + explore_reward + equip_reward + forag_reward
            # reward[i] = (sum(achv[i].values()) - sum(prev_achv[i].values()))/100.0
            # for key in self.keys:
            #     reward[i] += (self.agent_stage[i][key] - self.last_agent_stage[i][key])*10
        return reward

    def parse(self, achv):
        max_achv = {}
        total_reward = {i:0 for i in achv}
        for key in self.keys:
            max_achv[key] = max([achv[i][key] for i in achv])
        self.team_stage = self.update_team_stage(max_achv)
        self.agent_stage = self.update_agent_stage(achv)
        agent_reward = self.compute_agent_reward(self.prev_achv, achv)
        team_reward = self.compute_team_reward(self.max_prev_achv, max_achv)
        for i in achv:
            total_reward[i] = (1-self.team_spirit) * agent_reward[i] + self.team_spirit*team_reward
        self.last_team_stage = self.team_stage
        self.last_agent_stage = self.agent_stage
        self.prev_achv = achv
        self.max_prev_achv = max_achv
        return total_reward
    # def parse(self, prev_achv, achv):
    #     max_achv = {}
    #     total_reward = {i:0 for i in achv}
    #     for key in self.keys:
    #         max_achv[key] = max([achv[i][key] for i in achv])
    #     self.team_stage = self.update_team_stage(max_achv)
    #     self.agent_stage = self.update_agent_stage(achv)
    #     agent_reward = self.compute_agent_reward(self.prev_achv, achv)
    #     team_reward = self.compute_team_reward(self.max_prev_achv, max_achv)
    #     for i in achv:
    #         total_reward[i] = (1-self.team_spirit) * agent_reward[i] + self.team_spirit*team_reward
    #     self.last_team_stage = self.team_stage
    #     self.last_agent_stage = self.agent_stage
    #     self.prev_achv = achv
    #     self.max_prev_achv = max_achv

    #     reward = {
    #         i: (sum(achv[i].values()) - sum(prev_achv[i].values())) / 100.0
    #         for i in achv
    #     }
    #     return reward
    
