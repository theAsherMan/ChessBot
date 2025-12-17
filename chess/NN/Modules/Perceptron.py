import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from typing import Optional

class Perceptron(Module):
    def __init__(self, in_dim:int, out_dim:Optional[int] = None, drop_out:float = 0.5, residual:Optional[bool] = None):
        super().__init__()

        if out_dim is None:
            out_dim = in_dim
        
        if residual is None:
            residual = in_dim == out_dim
        assert((not residual) or (in_dim == out_dim), 'residuals cannot be applyed to data that is reshaped')
        self.is_residual = residual
        
        hid_dim = round((2/3)*in_dim+out_dim)
        
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop_out)
        )
    
    def forward(self, data:Tensor):
        return data+self.layers(data) if self.is_residual else self.layers(data)