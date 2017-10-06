#!/Users/aaron/Desktop/rl_practice/tf_venv/bin/python

#run this from the project directory containing 'final_data'
import os
import numpy as np 
from parsing import * 
import multiprocessing as mp

cwd = os.getcwd()
data_dir = os.path.join(cwd,'final_data')
dicom_dir = os.path.join(data_dir, 'dicoms')
cont_dir = os.path.join(data_dir, 'contourfiles')
link_dir = os.path.join(data_dir, 'link.csv')

#map the dicom to their respective contours

with open(link_dir, 'r') as dicom_links: 
	links = dicom_links.readlines()

# ['patient_id,original_id\n', 'SCD0000101,SC-HF-I-1\n', 'SCD0000201,SC-HF-I-2\n', 'SCD0000301,SC-HF-I-4\n', \
#  'SCD0000401,SC-HF-I-5\n', 'SCD0000501,SC-HF-I-6\n']
#dicom_to_contour = dict(item.strip().split(',') for item in links[1:]) 

contour_to_dicom_link = dict(item.strip().split(',')[::-1] for item in links[1:]) #ignore the header

def read_dicoms(d_dir = dicom_dir):
	"""
	returns a list of all dicom file paths
	"""
	all_dicoms = []
	dicom_sessions = os.listdir(d_dir) #all the individual imaging sessions have their own directories 
	for session in dicom_sessions:
	    sess_dir = os.path.join(d_dir, session)
	    sess_dicoms = os.listdir(sess_dir) #each session has a set of dicoms
	    for dicom in sess_dicoms:
	    	all_dicoms.append(os.path.join(sess_dir, dicom))
	return all_dicoms

def read_contours(c_dir = cont_dir):
	"""
	returns a list of all i-contour file paths
	"""
	all_contours =[]
	contour_sessions = os.listdir(c_dir)
	for session in contour_sessions:
		sess_dir = os.path.join(c_dir, session)
		sess_contour_dir = os.path.join(sess_dir, 'i-contours')
		sess_contours = os.listdir(sess_contour_dir)
		for contour in sess_contours:
			i_contour_dir = os.path.join(sess_contour_dir, contour)
			all_contours.append(i_contour_dir)
	return all_contours

# def dicom_to_contour(d_path, link_map = dicom_to_contour):
# 	"""
# 	Uses the Id links in the csv to map a dicom file path to contours
# 	returns the path to the contour file for a given DICOM 
# 	"""
# 	split_d_path = d_path.split('/')
# 	dicom_id = split_d_path[-2] #session id
# 	dicom_num = split_d_path[-1].split('.')[0] #image number 
# 	contour_id = link_map[dicom_id]   #associated contour session id 
#     contour_dir = os.path.join(cont_dir + '/i-contours', )    ..

def contour_to_dicom(c_path, link_map = contour_to_dicom_link):
	"""
    Linking the dicoms to the contours is pretty nasty. It's actually a lot easier to link the contours to the dicom. 
    Also, some dicoms don't have contours. 

    :param c_path: contour file path

    returns the path of the associated dicom file
	"""
	split_c_path = c_path.split('/')
	c_id = split_c_path[-3] 
	c_num = int(split_c_path[-1].split('-')[2]) #contour number associated with the dicom image
	if c_id in link_map:
	    dicom_id = link_map[c_id]   #associated dicom session
	else: 
		return ''
	d_dir = os.path.join(dicom_dir, '{0}/{1}.dcm'.format(dicom_id,c_num))
    # does this exist on the path? 
	if os.path.isfile(d_dir):
		return d_dir 
	else:
		return ''


def link_dicom_contour_mask(dicom_dir = dicom_dir, cont_dir = cont_dir): 
	""" 
	returns a tuple of DICOM image and its associated i-contour's boolean mask
	"""
	dicom_files = read_dicoms(dicom_dir)
	contour_files = read_contours(cont_dir)
	DICOMS, masks = [], []  ##these will eventually become the returned tuple
	for file in contour_files:
		# get all of the masks into the mask array. rows & columns of the dicom are 255/255
		# import ipdb
		# ipdb.set_trace()
		matching_dicom = contour_to_dicom(file, contour_to_dicom_link)
		if matching_dicom:
			dcm = parse_dicom_file(matching_dicom)
			height, width = dcm['height'], dcm['width']
			masks.append(poly_to_mask(parse_contour_file(file),width,height))
            #for every entry in the mask array, link it to the matching dicom file 
			DICOMS.append(dcm['pixel_data'])
	return np.stack(DICOMS), np.stack(masks)

#1 How did you verify that you are parsing the contours correctly?
    #I wrote a test case for rendering the mask for a mock polygon. I actually found that the provided code is sensitive to ordering
    #of coordinate pairs--which is probably an undesirable feature. Then I made sure that the enumeration of the number of files actually matched
    #the return value of my bash command 

#2. What changes did you make to the code, if any, in order to integrate it into our production code base?
	#I made it so that the parse_dicom_file function inserted image dimensions into the return dictionary that could be used for the 
	#mask generation function

#3. If the pipeline was going to be run on millions of images, and speed was paramount, how would you parallelize it to run as fast as possible?
	#I would  partition files to different executors based on the directory session id using the multiprocessing module. 
	#Then these executors can do the work of finding their partition's mapping of DICOM --> binary mask.
	#in the end, I would aggregate their results into a complete candidate set of paired inputs into the training pipeline. 
	#This is essentially like a mapreduce. You can scale to multiple machines, without changing the code much


#4. If this pipeline were parallelized, what kinds of error checking and/or safeguards, if any, would you add into the pipeline?
    #A few things:
        #I would want to make sure that files directories are strictly partitioned to different task threads that will not try to process the same file at once,
        #else we could get redundant data in the return tuple. (I don't think concurrent reads would actually be problematic)
        #I would have to make sure that within an epoch of training that no data indices would be reused across worker threads

