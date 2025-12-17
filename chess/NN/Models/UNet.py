import sys
sys.path.append('..')

from NN.Modules import ConvolutionalLayer

from .ResNet import ResNet

import torch
from torch.nn import Module,Sequential
from torch import Tensor

class UNet(Module):
    def __init__(self, channels:int, half_depth:int, sub_net_depth:int = 2):
        assert(half_depth > 0,f'UNet must have at least 1 layer.  requested UNet of half depth {half_depth}')
        super().__init__()
        self.is_base = half_depth == 1
        self.channels = channels
        self.down_pass_res_net = ResNet(channels, sub_net_depth)
        if not self.is_base:
            self.sub_net = _SubNet(channels=channels, half_depth=half_depth-1, sub_net_depth=sub_net_depth)
            self.compressor = ConvolutionalLayer(in_channels=channels*2, out_channels=channels, k_size=1)
            self.up_pass_res_net = ResNet(channels, sub_net_depth)
            

    def forward(self, data:Tensor):
        data = self.down_pass_res_net(data)
        if self.is_base:
            return data
        data = torch.cat((data,self.sub_net(data)),dim=1)
        data = self.compressor(data)
        data = self.up_pass_res_net(data)
        return data

class _SubNet(UNet):
    def __init__(self, channels, half_depth, sub_net_depth):
        self.down_sampler = ConvolutionalLayer(in_channels=channels, down_sample=True)
        super().__init__(channels=2*channels, half_depth=half_depth, sub_net_depth=sub_net_depth)
        self.up_sampler = ConvolutionalLayer(in_channels=2*channels, up_sample=True)
    
    def forward(self, data:Tensor):
        data = self.down_sampler(data)
        data = super().forward(data)
        data = self.up_sampler(data)
        return data