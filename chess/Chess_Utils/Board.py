import sys
sys.path.append('..')

import chess

from .CastlingRights import CastlingRights
from .Move import Move
from Maths import primes
from typing import Optional
from threading import Lock

class Board(chess.Board):
    def __init__(self, fen: Optional[str] = chess.STARTING_FEN, *, chess960: bool = False):
        super().__init__(fen=fen, chess960=chess960)
        self.lock = Lock()
    
    @property
    def legal_moves(self):
        for move in super().legal_moves:
            yield Move.fromSuper(move)

    def push(self, move:Move):
        class MoveHandler:
            def __init__(self, board:Board):
                self.board = board
            def __enter__(self):
                pass
            def __exit__(self, exc_type, exc_value, traceback):
                self.board.pop()
        if chess.square_rank(move.to_square) in [0,7]:
            if self.piece_at(move.to_square) == chess.PAWN:
                if not move.promotion:
                    raise AssertionError('cannot move pawn to back rank and not promote')
        super().push(move)
        return MoveHandler(self)

    def allPieces(self):
        for colour in chess.COLORS:
            for piece in chess.PIECE_TYPES:
                for tile in self.pieces(piece, colour):
                    yield tile,piece,colour

    def hasCastlingRight(self, castling_right:CastlingRights):
        match castling_right:
            case CastlingRights.WHITE_KINGSIDE:
                return self.has_kingside_castling_rights(chess.WHITE)
            case CastlingRights.WHITE_QUEENSIDE:
                return self.has_queenside_castling_rights(chess.WHITE)
            case CastlingRights.BLACK_KINGSIDE:
                return self.has_kingside_castling_rights(chess.BLACK)
            case CastlingRights.BLACK_QUEENSIDE:
                return self.has_queenside_castling_rights(chess.BLACK)
    


    def __hash__(self):
        hasher(self)

class _BoardHasher:
    def __init__(self):
        keys = []
        for tile in chess.SQUARES:
            for piece in chess.PIECE_TYPES:
                for colour in chess.COLORS:
                    keys.append((tile,piece,colour))
                    if chess.square_rank(tile) == 2 or chess.square_rank(tile) == 5:
                        keys.append((tile,'ep'))
        
        for right in CastlingRights:
            keys.append(right)
        
        self.mapping = {(key,prime) for key,prime in zip(keys,primes())}
    
    def __call__(self, board:Board):
        value = 1
        for tile,piece_type,colour in board.allPieces():
            value *= self.mapping[(tile,piece_type,colour)]
        ep_tile = board.ep_square
        if ep_tile:
            value *= self.mapping[(ep_tile,'ep')]
        for right in CastlingRights:
            if board.hasCastlingRight(right):
                value *= self.mapping[right]
        return value

hasher = _BoardHasher()