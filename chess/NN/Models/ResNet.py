import sys
sys.path.append('..')

from NN.Modules import ConvolutionalLayer
from torch import Tensor,nn
from torch.nn import Module

class ResNet(Module):
    def __init__(self, channels:int, depth:int):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.net = nn.Sequential(*[
            ConvolutionalLayer(channels, residual=True)
        ]*depth)
    
    def forward(self, data:Tensor):
        return self.net(data)