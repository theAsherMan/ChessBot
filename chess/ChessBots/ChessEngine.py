import sys
sys.path.append('..')

import torch
from torch import Tensor
import numpy as np
from abc import ABC, abstractmethod
from chess import Color,square,QUEEN,PAWN

from Chess_Utils import Board,ChessGraph,Move
from DSA import CircularStack
from Models import SimpleUNet
from NN.Devices import best_device

from typing import Optional

class ChessEngine(ABC):
    @abstractmethod
    def makeMove(self, board:Board):
        pass


class ZeroDepthEngine(ChessEngine):
    def __init__(self, save_loc:str):
        self.save_loc = save_loc
        try:
            self.model = SimpleUNet.load(save_loc)
        except FileNotFoundError:
            self.model = SimpleUNet(device=best_device, optimiser_type=torch.optim.Adam)
        self.boards:list[Board] = []
        self.moves:list[Move] = []
    
    def makeMove(self, board:Board):
        def extractMove(tensor:Tensor,stockastic:bool):
            if stockastic:
                array = tensor.numpy()
                array = array.flatten()
                array = array / np.sum(array)
                choice = np.random.choice(len(array), p=array)
                composition = np.unravel_index(choice, (8,8,8,8))
                return Move.recompose(*composition)
            else:
                pass
                #TODO
        turn = board.turn
        if turn:
            board = board.mirror()
        with torch.no_grad():
            pred = self.model.forward([board])
        pred = pred.squeeze()
        move = extractMove(pred, stockastic=True)
        if turn:
            move.apply_mirror()
        return move
        

    def new_game(self):
        self.start_pos = None
        self.move_record = []
    
    def train(self, winner:Optional[Color]):
        if winner is None:return

        move_evals:list[np.array] = []
        for board,move in zip(self.boards,self.moves):
            
            array = np.zeros((8,8,8,8))
            array[*move.decompose()] = 1
            move_evals.append(array)
        
        self.model.backward(boards=self.boards, move_eval=move_evals)
        self.model.save(self.save_loc)