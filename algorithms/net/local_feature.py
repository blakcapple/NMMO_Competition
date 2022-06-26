import torch 
import torch.nn as nn
from algorithms.net.cnn_layer_local import CNNBase
from algorithms.net.mlp_layer import MLPBase

class LocalNet(nn.Module):
    """
    The Net used for extracting local information for every agents
    """
    def __init__(self, config):
        
        super().__init__()
        
        self.agent_net = MLPBase(17, 64, 2, use_feature_normalization=False)
        self.map_net = CNNBase(in_channels=37)
        self.friend_net = MLPBase(7*17, 128, 2, use_feature_normalization=False)

    def forward(self, input:dict):
        """
        input consists of following part:
        agent_vector: shape(n, 8, 17)
        friend_vector: shape(n, 8, 7, 17)
        local_map: shape(n, 8, 17,15,15)
        out: shape(n, 8, 64+512+128=704)
        """
        batch_size = input['agent_vector'].shape[0]
        agent_vector = input['agent_vector']
        friend_vector = input['team_vector']
        local_map = input['local_map']
        local_map = local_map.view(batch_size*8, *local_map.shape[2:])
        
        agent_feature = self.agent_net(agent_vector)
        map_feature = self.map_net(local_map)
        map_feature = map_feature.view(batch_size, 8, -1)
        friend_feature = self.friend_net(friend_vector)
        local_feature = torch.concat([agent_feature, map_feature, friend_feature], dim=2)

        return local_feature
