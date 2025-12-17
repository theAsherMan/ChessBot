import threading

class Lock(type(threading.Lock())):
    def __enter__(self):
        self.acquire()
    
    def __exit__(self, type, value, traceback):
        self.release()
        return False