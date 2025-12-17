from collections.abc import Iterable

class CircularStack[T]:
    def __init__(self, max_length:int = -1, initual_elements:Iterable[T] = []):
        self.list = list(initual_elements)
        self.max_length = max_length
    
    def add(self, element:T):
        if len(self.list) == self.max_length:
            self.list.pop()
        self.list.insert(0, element)
    
    def __iter__(self):
        return iter(self.list)
    
    def pop(self, idx:int):
        self.list.pop(idx)
    
    def remove(self, element:T):
        self.list.remove(element)