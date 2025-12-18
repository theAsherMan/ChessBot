import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from typing import Optional
from math import floor

class ConvolutionalLayer(Module):
    def __init__(self, in_channels:int, out_channels:Optional[int] = None, k_size:Optional[int] = None, up_sample:bool = False, down_sample:bool = False, residual:Optional[bool] = None):
        
        if up_sample and down_sample:raise ValueError('convolutional layer cannot up-sample and down-sample')

        if k_size is None:
            if up_sample:
                k_size = 4
            else:
                k_size = 3
        
        if out_channels is None:
            if up_sample:
                out_channels = round(in_channels/2)
            elif down_sample:
                out_channels = in_channels*2
            else:
                out_channels = in_channels
        
        if residual is None:
            residual = not (up_sample or down_sample) and in_channels == out_channels
        
        assert((not residual) or not (up_sample or down_sample), 'residuals cannot be applyed to data that is reshaped')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_size = k_size
        self.is_residual = residual
        
        super().__init__()
        self.layers = nn.Sequential()
        if up_sample:
            self.layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(k_size,k_size),
                    stride=2,
                    padding=int(k_size/2-1)
                )
            )
        elif down_sample:
            self.layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(k_size,k_size),
                    stride=2,
                    padding=floor(k_size/2)
                )
            )
        else:
            self.layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(k_size,k_size),
                    stride=1,
                    padding=floor(k_size/2)
                )
            )
        self.layers.append(nn.GELU())
    
    def forward(self, data:Tensor):
        return data+self.layers(data) if self.is_residual else self.layers(data)