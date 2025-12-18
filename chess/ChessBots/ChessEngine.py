import sys
sys.path.append('..')

import torch
from torch import Tensor
import numpy as np
from abc import ABC, abstractmethod
from chess import Color,square,QUEEN,PAWN,square_rank,Outcome,Termination

from Chess_Utils import Board,ChessGraph,Move
from DSA import CircularStack
from .Models import SimpleUNet
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
    
    def makeMove(self, board:Board):
        def generateMoveMask(board:Board):
            mask = torch.full((8,8,8,8), float('-inf'), device=best_device)
            for move in board.legal_moves:
                mask[*move.decompose()] = 0
            return mask
        
        def extractMove(move_logits:Tensor,move_mask:Tensor,stockastic:bool):
            move_logits = move_logits.flatten()
            move_mask = move_mask.flatten()
            tensor = move_logits + move_mask
            if stockastic:
                tensor = torch.nn.functional.softmax(tensor, dim=0)
                array = tensor.cpu().numpy()
                choice = np.random.choice(len(array), p=array)
                composition = np.unravel_index(choice, (8,8,8,8))
                return Move.recompose(*composition)
            else:
                array = tensor.cpu().numpy()
                choice = np.argmax(array)
                composition = np.unravel_index(choice, (8,8,8,8))
                return Move.recompose(*composition)
        
        def getPromotion(board:Board, move:Move):
            from_square = move.from_square
            piece = board.piece_at(from_square)
            to_square = move.to_square
            to_rank = square_rank(to_square)
            if piece == PAWN:
                if to_rank == 0 or to_rank == 7:
                    return QUEEN
            return None
        

        turn = board.turn
        if turn:
            board = board.mirror()
        with torch.no_grad():
            pred = self.model.forward([board])
        pred = pred.squeeze()
        mask = generateMoveMask(board)
        move = extractMove(pred, mask, stockastic=True)
        promotion = getPromotion(board, move)
        move.promotion = promotion
        if turn:
            move.apply_mirror()
        return move
        

    def new_game(self):
        self.start_pos = None
        self.move_record = []
    
    def train(self, boards:list[Board], moves:list[Move], outcome:Outcome):
        if outcome.termination not in [Termination.CHECKMATE,Termination.STALEMATE]:return
        winner = outcome.winner

        move_evals:list[np.array] = []
        for board,move in zip(boards, moves):
            
            array = np.zeros((8,8,8,8))
            value = 1 if board.turn == winner else 0
            value = 0.5 if winner is None else value
            array[*move.decompose()] = value
            move_evals.append(array)
        
        self.model.backward(boards=self.boards, move_eval=move_evals)
        self.model.save(self.save_loc)