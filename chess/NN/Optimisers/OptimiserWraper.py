import sys
sys.path.append('..')

from NN.Devices import available_devices
import torch

if torch.device('TPU') in available_devices:
    import torch_xla.core.xla_model as xm

class OptimiserWraper:
    def __init__(self, optimiser:torch.optim.Optimizer, device:torch.device):
        self.optim = optimiser
        self.device = device
    
    def backward(self, loss:torch.Tensor):
        loss.backward()
        self.step()
    
    def step(self, zero_grad:bool = True):
        if self.device == torch.device('tpu'):
            xm.optimizer_step(self.optim)
        else:
            self.optim.step()
        if zero_grad:
            self.zero_grad()
    
    def zero_grad(self):
        self.optim.zero_grad()