import chess
from typing import Optional

from dataclasses import fields, replace

class Move(chess.Move):
    @classmethod
    def recompose(cls, from_rank, from_file, to_rank, to_file, promotion:Optional[chess.PieceType] = None):
        return Move(from_square=chess.square(from_file,from_rank), to_square=chess.square(to_file,to_rank),promotion=promotion)

    @classmethod
    def fromSuper(cls, source:chess.Move):
        return Move(**{f.name: getattr(source, f.name) for f in fields(source)})

    def decompose(self):
        from_square = self.from_square
        to_square = self.to_square

        from_rank = chess.square_rank(from_square)
        from_file = chess.square_file(from_square)

        to_rank = chess.square_rank(to_square)
        to_file = chess.square_file(to_square)

        return from_rank,from_file,to_rank,to_file
    
    def mirror(self):
        move = replace(self)
        move.apply_mirror()
        return move

    def apply_mirror(self):
        from_rank,from_file,to_rank,to_file = self.decompose()
        
        from_rank = 7-from_rank
        to_rank = 7-to_rank

        self.from_square = chess.square(from_file, from_rank)
        self.to_square = chess.square(to_file, to_rank)