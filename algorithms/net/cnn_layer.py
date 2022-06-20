import torch.nn as nn
from algorithms.utils.util import init
import torch 
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNLayer(nn.Module):
    def __init__(self, use_orthogonal, use_ReLU):
        super(CNNLayer, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=17,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1)), nn.ReLU(),
            init_(nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1)), nn.ReLU(),
            init_(nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1)), nn.ReLU(),
            init_(nn.Conv2d(in_channels=32,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1)), nn.ReLU(),
            init_(nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1)), nn.ReLU())
        self.core = (nn.Linear(16*15*15, 256))

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.core(x))
        return x
        
class CNNBase(nn.Module):
    def __init__(self):
        super(CNNBase, self).__init__()

        self._use_orthogonal = True
        self._use_ReLU = True

        self.cnn = CNNLayer(self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        x = self.cnn(x)
        return x