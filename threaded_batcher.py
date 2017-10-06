#!/Users/aaron/Desktop/rl_practice/tf_venv/bin/python
import threading
import numpy as np

class Threadsafe_Batcher(object):
    """
    Takes a range of indices and makes thread-safe random batching by serializing call to the `next` method.
    """
    _instance = None
    _lock = threading.RLock()
    def __init__(self, num_indices, batch_size):
        self.num_indices = num_indices
        self.batch_size = batch_size
        self.lock = threading.Lock()
        self.flush()
        Threadsafe_Batcher._instance = self
    
    def flush(self):
        self.curr_idx = 0
        self.indices = np.random.permutation(self.num_indices) #random array of the range of the indexes

    @staticmethod
    def GetInstance():
        if (Threadsafe_Batcher._instance == None):
            Threadsafe_Batcher._lock.acquire()
            if (Threadsafe_Batcher._instance == None):
                Threadsafe_Batcher._instance = Threadsafe_Batcher(self.num_indices, self.batch_size)
            Threadsafe_Batcher._lock.release()
        return Threadsafe_Batcher._instance
   
    @staticmethod
    def Reset():
        Threadsafe_Batcher._instance = None

    def __next__(self):
        with self.lock:
            if self.curr_idx >= self.num_indices:
                #start at the beginning
                self.flush()
            #choose the minimum of the batch size and the remaining elements in the collection
            num_sample = min(self.batch_size, self.num_indices - self.curr_idx)
            sample_indexes = self.indices[self.curr_idx:self.curr_idx + num_sample]
            self.curr_idx += num_sample
            return sample_indexes