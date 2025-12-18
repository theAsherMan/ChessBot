import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from typing import Optional

class Perceptron(Module):
    def __init__(self, in_dim:int, out_dim:Optional[int] = None, drop_out:float = 0.5, residual:Optional[bool] = None):
        super().__init__()

        self.drop_out = drop_out

        if out_dim is None:
            out_dim = in_dim
        
        if residual is None:
            residual = in_dim == out_dim
        assert((not residual) or (in_dim == out_dim), 'residuals cannot be applyed to data that is reshaped')
        self.is_residual = residual
        
        hid_dim = round((2/3)*in_dim+out_dim)

        self.in_layer = nn.Linear(in_dim, hid_dim)
        self.out_layer = nn.Linear(hid_dim, out_dim)
    
    def forward(self, data:Tensor):
        data = self.in_layer(data)
        data = nn.functional.gelu(data)
        data = self.out_layer(data)
        data = nn.functional.gelu(data)
        data = nn.functional.dropout(data, p=self.drop_out)
        return data