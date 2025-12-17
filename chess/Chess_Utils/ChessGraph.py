from __future__ import annotations

import sys
sys.path.append('..')


from .dataclasses import dataclass, field, fields

from .Board import Board
from .Move import Move

@dataclass
class Edge:
    graph:ChessGraph
    move:Move
    destination:Node
    value:float = float('nan')


@dataclass
class Node:
    graph:ChessGraph
    board:Board
    value:float = float('nan')
    edges_from:dict[Move,Edge] = field(default_factory=dict, repr=False)
    parents:set[Node] = field(default_factory=set, repr=False)

    def __hash__(self):
        return hash(self.board)
    
    def __eq__(self, value):
        if type(self) != type(value):return False
        if {f.name: getattr(self, f.name) for f in fields(self)} != {f.name: getattr(value, f.name) for f in fields(value)}:return False
        return True

    def getEdge(self, move:Move):
        if move in self.edges_from:
            return self.edges_from[move]
        with self.board.push(move):
            destination = self.graph.getNode(self.board)
            destination.parents.add(self)
            new_edge = Edge(self.graph, move, destination)
            self.edges_from[move] = new_edge
            return new_edge
    
    def setValue(self, value:float):
        self.value = value
        self.graph.backProp(self)

class ChessGraph:
    def __init__(self, start_position:Board):
        self.start_position = start_position
        self.nodes:dict[int,Node] = {}
    
    def getNode(self, position:Board):
        hash_value = hash(position)
        if hash_value in self.nodes:
            return self.nodes[hash_value]
        new_node = Node(self, position.copy())
        self.nodes[hash_value] = new_node
        return new_node