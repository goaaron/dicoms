import numpy as np 
from threaded_batcher import Threadsafe_Batcher
from parsing_utils import * 
import math
from threading import Thread

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs, Verbose)
        self._return = None
    def run(self):
        if self._Thread__target is not None:
            self._return = self._Thread__target(*self._Thread__args,
                                                **self._Thread__kwargs)
    def join(self):
        Thread.join(self)
        return self._return

class ML_Pipeline(object): 
	"""
	The pipeline which will feed (DICOM,mask) tuples to a model
	"""
	def __init__(self, epochs, DICOMS, mask, batch_size = 8):
		self.batcher = Threadsafe_Batcher(DICOMS.shape[0], batch_size)
		self.DICOMS = DICOMS
		self.mask = mask
		self.batch_size = batch_size
		self.epochs = epochs

	def batch_data(self):
	    #how to make this an asynchronous yield? 
	    batch_indices = self.batcher.__next__()
	    return self.DICOMS[batch_indices], self.mask[batch_indices]

	def async_batch(self): 
		lstProcess = []
		gens = []
		for i in range(int(math.ceil(self.DICOMS.shape[0] // self.batch_size))):
			thread = ThreadWithReturnValue(target=self.batch_data)
			lstProcess.append(thread)
			thread.start()
		for thread in lstProcess:
			gens.append(thread.join())
		for dicom, mask in gens:
			yield dicom, mask 
			