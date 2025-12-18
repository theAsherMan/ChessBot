import sys
sys.path.append('..')

from NN.Models import UNet,ResNet
from NN.Modules import Perceptron,ConvolutionalLayer
from NN.Optimisers import OptimiserWraper

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn as nn
from math import log2
import numpy as np
from typing import Optional
import pickle
import os
from pathlib import Path

from chess import COLORS, PIECE_TYPES, SQUARES, square_rank, square_file

from Chess_Utils import Board,CastlingRights,Move

class Tensorfier:
    def __init__(self):
        self.attributes = []
        for colour in COLORS:
            for piece in PIECE_TYPES:
                self.attributes.append((colour,piece))
        self.attributes.append('ep')
        for right in CastlingRights:
            self.attributes.append(right)
    
    def __len__(self):
        return len(self.attributes)
    
    def toBatch(self, boards:list[Board]):
        return torch.stack([self.toTensor(board) for board in boards]).detach().requires_grad_()
    
    def toTensor(self, board:Board):
        def tensorfyPosition(board:Board):
            tensor = torch.zeros((len(self),8,8))
            for square in SQUARES:
                piece = board.piece_at(square)
                if piece:
                    colour = board.color_at(square)
                    rank = square_rank(square)
                    file = square_file(square)
                    encoding = [1 if attribute == (colour,piece) else 0 for attribute in self.attributes]
                    tensor[:,rank,file] = torch.Tensor(encoding)
            ep_square = board.ep_square
            if ep_square:
                encoding = [1 if attribute == 'ep' else 0 for attribute in self.attributes]
                rank = square_rank(ep_square)
                file = square_file(ep_square)
                tensor[:,rank,file] = torch.Tensor(encoding)
            return tensor

        def tensorfyRights(board:Board):
            tensor = torch.zeros((len(self)))
            for right in CastlingRights:
                if board.hasCastlingRight(right):
                    sub_tensor = [1 if attribute == right else 0 for attribute in self.attributes]
                    sub_tensor = torch.tensor(sub_tensor)
                    tensor += sub_tensor
            return tensor

        position_tensor = tensorfyPosition(board)
        rights_tensor = tensorfyRights(board)
        tensor = position_tensor.permute(1,2,0)+rights_tensor
        tensor = tensor.permute(2,0,1)
        return tensor.detach().requires_grad_()

tensorfier = Tensorfier()

class SimpleUNet(Module):
    def __init__(self, device:torch.device, optimiser_type:type[torch.optim.Optimizer]):
        super().__init__()
        self.device = device
        self.channels = len(tensorfier)
        self.model = UNet(
            channels=len(tensorfier),
            half_depth=log2(8),
            sub_net_depth=5
        ).to(self.device)
        self.head = Perceptron(len(tensorfier)*8*8,8*8*8*8, drop_out=0.0).to(self.device)
        self.optim = OptimiserWraper(optimiser_type(self.parameters()), self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.eval()
    
    def forward(self, boards:list[Board]) -> Tensor:
        data = tensorfier.toBatch(boards).to(self.device)
        data = self.model(data)
        data = torch.flatten(data, start_dim=1)
        data = self.head(data)
        return data
    
    def backward(self, boards:list[Board], move_eval:list[np.ndarray]):
        self.train()
        move_preds = self.forward(boards)
        move_eval = torch.tensor(np.array(move_eval)).to(self.device)
        move_loss = self.loss_fn(move_preds, move_eval)
        self.optim.backward(move_loss)
        self.eval()
    
    def save(self, file_url:str):
        path = Path(file_url)
        os.makedirs(path.parent, exist_ok=True)
        with open('file_url','wb'):
            pickle.dump(self, file_url)
    
    @classmethod
    def load(cls, file_url:str):
        with open(file_url, 'rb') as file:
            return pickle.load(file)