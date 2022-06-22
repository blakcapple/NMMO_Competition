import torch 
import torch.nn as nn
from algorithms.net.mlp_layer import MLPBase

class GlobalNet(nn.Module):
    """
    The Net used for extracting global information for team
    """
    def __init__(self, config):

        super().__init__()
        self.npc_net = MLPBase(17, 64, layer_N=1)
        self.enermy_net = MLPBase(17, 64, layer_N=1)
        self.team_net = MLPBase(8*17, 128, layer_N=2)
        self.time_net = MLPBase(1, 32, layer_N=1)

    def forward(self, input):
        """
        input consists of following part:
        npc_vector: shape(n, 20, 17)
        enemy_vector: shape(n, 20, 17)
        team_vector: shape(n, 8*17)
        time: shape(n, 1)
        out: shape(n, 8, 64+64+128+32=288)
        """
        npc_vector = input['npc_vector']
        enemy_vector = input['enemy_vector']
        team_vector = input['team_vector']
        time = input['time']

        npc_feature = self.npc_net(npc_vector).max(1).values
        enemy_feature = self.enermy_net(enemy_vector).max(1).values
        team_feature = self.team_net(team_vector)
        time_feature = self.time_net(time)
        global_feature = torch.concat([npc_feature, enemy_feature, team_feature, time_feature], dim=1)
        global_feature = global_feature.unsqueeze(1)
        global_feature = global_feature.repeat(1, 8, 1)
        return global_feature
        