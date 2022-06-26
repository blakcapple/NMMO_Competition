import torch.nn as nn
from algorithms.utils.util import init
import torch 
import torch.nn.functional as F

class Residual_Block(nn.Module):
    
    def __init__(self, planes, stride=1, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                        padding=1, bias=True) # before bn, bias in cnn is helpless 
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                        padding=1, bias=True)
        # self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x 

        out = self.conv1(x)
        out = F.relu((out))
        out = self.conv2(out)
        out += residual 
        out = F.relu(out)

        return out 

class CNNLayer(nn.Module):
    def __init__(self, in_channels, use_orthogonal, use_ReLU):
        super(CNNLayer, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        
        self.block_num = 3

        self.cnn = nn.Sequential(
                    nn.Conv2d(in_channels, 32, kernel_size=1, stride=3,
                        padding=1), self.max_pool,nn.ReLU())
        self.block_1 = Residual_Block(planes=32)
        self.block_2 = Residual_Block(planes=32)

        # with torch.no_grad():
        #     dummy_ob = torch.ones(1, in_channels, 128, 128).float()
        #     dummy_ob = self.cnn(dummy_ob)
        #     dummy_ob = self.block_1(dummy_ob)
        #     dummy_ob = self.block_2(dummy_ob)
        #     dummy_ob = self.avg_pool(dummy_ob)
        #     dummy_ob = torch.flatten(dummy_ob, start_dim=1)
        #     n_flatten = dummy_ob.shape[1]
        self.core = (nn.Linear(3872, 512))

    def forward(self, x):
        x = self.cnn(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.core(x))
        return x
        
class CNNBase(nn.Module):
    def __init__(self, in_channels):
        super(CNNBase, self).__init__()

        self._use_orthogonal = True
        self._use_ReLU = True

        self.cnn = CNNLayer(in_channels, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        x = self.cnn(x)
        return x