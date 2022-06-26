import torch 
import torch.nn as nn
from algorithms.net.cnn_layer_global import CNNBase
from algorithms.net.mlp_layer import MLPBase

class GlobalNet(nn.Module):
    """
    The Net used for extracting global information for team
    """
    def __init__(self, config):

        super().__init__()
        self.map_net = CNNBase(11)
        self.time_net = MLPBase(1, 32, layer_N=1)

    def forward(self, input):
        """
        input consists of following part:
        npc_vector: shape(n, 20, 17)
        enemy_vector: shape(n, 20, 17)
        team_vector: shape(n, 8*17)
        time: shape(n, 1)
        out: shape(n, 8, 32+512=544)
        """
        global_map = input['global_map']
        time = input['time']

        map_feature = self.map_net(global_map)
        time_feature = self.time_net(time)
        global_feature = torch.concat([map_feature, time_feature], dim=1)
        global_feature = global_feature.unsqueeze(1)
        global_feature = global_feature.repeat(1, 8, 1)
        return global_feature
        