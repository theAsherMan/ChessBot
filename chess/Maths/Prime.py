import sys
sys.path.append('..')

from threading import Lock

class _Prime_private:
    lock = Lock()
    array = [2]

    @classmethod
    def extend(cls):
        cls.extendToIdx(len(cls.array))

    @classmethod
    def extendToIdx(cls, idx):
        with cls.lock:
            while len(cls.array) <= idx:
                cls.extendAsync()
    
    @classmethod
    def extendAsync(cls):
        candidate = cls.array[-1] + 1
        while isComposite(candidate):
            candidate += 1
        cls.array.append(candidate)
    
    @classmethod
    def decompositionFactors(cls, number):
        for prime in primes:
            if number == 1:
                break
            while number%prime == 0:
                yield prime
                number /= prime

def primeAtIdx(idx:int):
    if len(_Prime_private.array) <= idx:
        _Prime_private.extendToIdx(idx)

def isComposite(number:int):
    limit = number**0.5
    for prime in _Prime_private.array:
        if number % prime == 0:
            return False
        if prime > limit:
            return True
    raise AssertionError(f'all calculated primes were iterated without sqrt being reached or primality found.\nnumber was:{number}\nsqrt:{limit}\nprimes:{_Prime_private.array}')

def primeDecomposition(number:int):
    if number < 2:raise ValueError('numbers less than 2 cannot by prime or composite')
    return [factor for factor in _Prime_private.decompositionFactors(number)]

@property
def primes():
    idx = 0
    while True:
        yield primeAtIdx(idx)
        idx += 1