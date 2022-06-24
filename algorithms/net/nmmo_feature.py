import torch 
import torch.nn as nn
from algorithms.net.global_feature import GlobalNet
from algorithms.net.local_feature import LocalNet
from algorithms.net.mlp_layer import MLPBase

class NMMONet(nn.Module):

    def __init__(self, config):
        
        super().__init__()
        self.local_net = LocalNet(config)
        self.global_net = GlobalNet(config)
        self.linear_net = MLPBase(704, 512, layer_N=1)

    def forward(self, input):
        
        local_obs = input['local_obs']
        global_obs = input['global_obs']

        local_feature = self.local_net(local_obs) # (n, 8, 416)
        global_feature = self.global_net(global_obs) # (n, 8, 288)
        assert(len(global_feature.shape)==3), print(global_feature.shape)
        assert(global_feature.shape[0]==local_feature.shape[0]), print(global_feature.shape, local_feature.shape)
        full_feature = torch.concat([local_feature, global_feature], dim=2)
        full_feature = self.linear_net(full_feature)
        full_feature = full_feature.view(-1, *full_feature.shape[2:]) # (n*8, 512)
        return full_feature
