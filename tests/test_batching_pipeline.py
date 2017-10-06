#!/Users/aaron/Desktop/rl_practice/tf_venv/bin/python
import os
import numpy as np
from unittest import TestCase
from .. import ml_pipeline  
from .. import threaded_batcher
from .. import parsing_utils

DICOMS, mask = parsing_utils.link_dicom_contour_mask()
pipeline = ml_pipeline.ML_Pipeline(2, DICOMS, mask, 8)
async_yield = pipeline.async_batch()

print next(async_yield)
class TestThreadedBatcher(TestCase): 
	def test_iter(self): 
		batches = set()
		batcher = threaded_batcher.Threadsafe_Batcher(20, 7)
		for i in range(2):
			batch =batcher.__next__()
			print batch
			self.assertTrue(batches.intersection(batch.tolist()) == set())
			self.assertTrue(len(batch) == 7)
			batches = batches.union(batch.tolist())
		last_batch = batcher.__next__()
		print last_batch
		self.assertTrue(batches.intersection(last_batch.tolist()) == set())
		self.assertTrue(len(last_batch) == 6) #choose the minimum of the remaining elements and the batch size
		batches = batches.union(last_batch.tolist())
		self.assertTrue(len(batches) == 20)

	"""
    [16 19  3 15  6  0  5]
	[17 10  2  7 13  9 12]
	[ 8  1 18  4 11 14]
    """

class TestMLPipeline(TestCase): 
	def test_batch(self):
		pass
