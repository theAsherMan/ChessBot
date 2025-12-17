import threading

class Lock(threading.Lock):
    def __enter__(self):
        self.acquire()
    
    def __exit__(self, type, value, traceback):
        self.release()
        return False