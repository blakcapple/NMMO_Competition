import torch 
import torch.nn as nn
from algorithms.net.global_feature import GlobalNet
from algorithms.net.local_feature import LocalNet

class NMMONet(nn.Module):

    def __init__(self, config):

        self.local_net = LocalNet(config)
        self.global_net = GlobalNet(config)

    def forward(self, input):
        
        local_obs = input['local_obs']
        global_obs = input['global_obs']

        local_feature = self.local_net(local_obs)
        global_feature = self.global_net(global_obs)
        full_feature = torch.concat([local_feature, global_feature], dim=2)

        return full_feature
