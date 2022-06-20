import torch 
import torch.nn as nn
from algorithms.net.cnn_layer import CNNBase
from algorithms.net.mlp_layer import MLPBase
from algorithms.net.atten_layer import SelfAtten

class LocalNet(nn.Module):
    """
    The Net used for extracting local information for every agents
    """
    def __init__(self, config):
        
        super().__init__()

        self.agent_net = SelfAtten(17, query_dim=64, embed_dim=32, atten_head=2)
        self.map_net = CNNBase()
        self.attack_net = MLPBase(20*12, 128, 2, use_feature_normalization=True)

    def forward(self, input:dict):
        """
        input consists of following part:
        agent_vector: shape(n, 8, 17)
        local_map: shape(n, 8, 17*15*15)
        attack_vector: shape(n, 8, 20*12)
        agent_mask: shape(n, 8)
        out: shape(n, 8, 32+256+128)
        """

        agent_vector = input['agent_vector']
        local_map = input['local_map']
        attack_vector = input['attack_vector']
        agent_mask = input['agent_mask']
        
        agent_feature = self.agent_net(agent_vector, agent_mask)
        map_feature = self.map_net(local_map)
        attack_feature = self.attack_net(attack_vector)

        local_feature = torch.concat([agent_feature, map_feature, attack_feature], dim=2)

        return local_feature
