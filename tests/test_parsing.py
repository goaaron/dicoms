#!/Users/aaron/Desktop/rl_practice/tf_venv/bin/python
import os
import numpy as np
from unittest import TestCase
from .. import parsing_utils
from .. import parsing

class Parsing_Test(TestCase):
	def test_read_dicoms(self):
		"""
		there are a total of 1140 dicoms in final_data
		ls -l SCD*/* | wc -l
		"""
		dicoms = parsing_utils.read_dicoms()
		self.assertTrue(len(dicoms) == 1140)

	def test_read_contours(self): 
		"""
		there are a total of 96 i-contours in final_data
		ls SC-HF-I-*/i-contours/* | wc -l
		"""
		i_contours = parsing_utils.read_contours()
		self.assertTrue(len(i_contours) == 96)

	def test_contour_to_dicom(self): 
		"""
		i would create a mock file and do the pattern matching with a predefined link dict
		"""
		pass

	def test_link_dicom_contour_mask(self):
		pass

	def test_parse_dicom_file(self):
		"""
		some dicom files may be corrupt
		"""
		dicoms = parsing_utils.read_dicoms()
		for dcm in dicoms: 
			self.assertTrue(parsing.parse_dicom_file(dcm) is not None)

	def test_parse_contour_file(self):
		"""
		generates a fake contour file and parses it into coordinates
		"""
		contour = "120 137\n120 137.0"
		with open('tmp.txt', 'wb') as f:
			f.write(contour)
		f.close()
		coords_list = parsing.parse_contour_file(os.path.join(os.getcwd(), 'tmp.txt'))
		self.assertTrue(coords_list[0] == (120,137))
		self.assertTrue(coords_list[-1] == (120,137))

	def test_poly_to_mask(self):
		"""
		the order in which a set of coordinates is parsed into a polygon mask will actually change the mask. this should not be the case
		"""
		#coords = [(0,0), (5,5), (0,5), (5,0)] NOT WORKING
		coords = [(0,0), (0,5), (5,5), (5,0)] #WORKING
		polygon = parsing.poly_to_mask(coords,10,10)
		np_mask = np.zeros((10,10), dtype=bool)
		np_mask[1:5,1:5] = True
		self.assertTrue(np.all(polygon == np_mask)) #mask for the square 




