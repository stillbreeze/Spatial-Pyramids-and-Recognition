import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import matplotlib.pyplot as plt
import util
import random

import os
import os.path
from scipy.ndimage import gaussian_filter, gaussian_laplace
from multiprocessing import Pool
from scipy.spatial.distance import cdist


def extract_filter_responses(image):
	'''
	Extracts the filter responses for the given image.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	[output]
	* filter_responses: numpy.ndarray of shape (H,W,3F)
	'''
	# Channel dimension check
	if image.ndim == 1:
		image = np.stack((image,)*3, -1)
	elif image.shape[-1] == 4:
		image = image[:,:,:3]

	# Image normalization check
	image_min = image.min(axis=(0,1),keepdims=True)
	image_max = image.max(axis=(0,1),keepdims=True)
	image = (image-image_min)/(image_max-image_min)

	# Variable initialization and color space conversion
	scale_set = [1,2,4,8,8*np.sqrt(2)]
	labimage = skimage.color.rgb2lab(image)
	filter_response = np.array([], dtype=labimage.dtype).reshape(labimage.shape[0], labimage.shape[1], 0)

	# Gaussian filter
	for scale in scale_set:
		response_l = gaussian_filter(labimage[:,:,0], scale)
		response_a = gaussian_filter(labimage[:,:,1], scale)
		response_b = gaussian_filter(labimage[:,:,2], scale)
		response = np.stack((response_l,response_a,response_b), -1)
		filter_response = np.concatenate([filter_response, response], -1)

	# Laplacian of Gaussian filter
	for scale in scale_set:
		response_l = gaussian_laplace(labimage[:,:,0], scale)
		response_a = gaussian_laplace(labimage[:,:,1], scale)
		response_b = gaussian_laplace(labimage[:,:,2], scale)
		response = np.stack((response_l,response_a,response_b), -1)
		filter_response = np.concatenate([filter_response, response], -1)

	# Gaussian derivative x direction filter
	for scale in scale_set:
		response_l = gaussian_filter(labimage[:,:,0], scale, order=[1,0])
		response_a = gaussian_filter(labimage[:,:,1], scale, order=[1,0])
		response_b = gaussian_filter(labimage[:,:,2], scale, order=[1,0])
		response = np.stack((response_l,response_a,response_b), -1)
		filter_response = np.concatenate([filter_response, response], -1)

	# Gaussian derivative y direction filter
	for scale in scale_set:
		response_l = gaussian_filter(labimage[:,:,0], scale, order=[0,1])
		response_a = gaussian_filter(labimage[:,:,1], scale, order=[0,1])
		response_b = gaussian_filter(labimage[:,:,2], scale, order=[0,1])
		response = np.stack((response_l,response_a,response_b), -1)
		filter_response = np.concatenate([filter_response, response], -1)

	return filter_response


def get_visual_words(image,dictionary):
	'''
	Compute visual words mapping for the given image using the dictionary of visual words.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	
	[output]
	* wordmap: numpy.ndarray of shape (H,W)
	'''

	filter_responses = extract_filter_responses(image)
	h, w, f = filter_responses.shape
	flattened_filters = filter_responses.reshape(h*w,f)
	distance_matrix = cdist(flattened_filters, dictionary)
	cluster_index = distance_matrix.argmin(axis=-1)
	cluster_index = cluster_index.reshape(h,w)
	return cluster_index


def compute_dictionary_one_image(args):
	'''
	Extracts random samples of the dictionary entries from an image.
	This is a function run by a subprocess.

	[input]
	* i: index of training image
	* alpha: number of random samples
	* image_path: path of image file
	* time_start: time stamp of start time

	[saved]
	* sampled_response: numpy.ndarray of shape (alpha,3F)
	'''
	i, alpha, image_path, temp_filter_path = args
	temp_file_name = temp_filter_path + str(i) + '.npy'
	if not os.path.exists(temp_file_name):
		image = skimage.io.imread(image_path)
		image = image.astype('float')/255
		filter_responses = extract_filter_responses(image)
		h, w, f = filter_responses.shape
		flattened_filters = filter_responses.reshape(h*w,f)
		random_filters = np.random.permutation(flattened_filters)[:alpha,:]
		np.save(temp_file_name, random_filters)



def compute_dictionary(num_workers=2):
	'''
	Creates the dictionary of visual words by clustering using k-means.

	[input]
	* num_workers: number of workers to process in parallel
	
	[saved]
	* dictionary: numpy.ndarray of shape (K,3F)
	'''

	K = 150
	alpha = 300
	temp_filter_path = '../data/temp_filters/'
	train_data = np.load('../data/train_data.npz')
	if not os.path.exists(temp_filter_path):
		os.makedirs(temp_filter_path)
	arglist = []
	for img_details in train_data['image_names']:
		img_index = str(img_details[0].split('/')[1].split('.')[0])
		img_path = str('../data/' + img_details[0])
		args = (img_index, alpha, img_path, temp_filter_path)
		arglist.append(args)

	p = Pool(num_workers)
	p.map(compute_dictionary_one_image, arglist)

	filters = []
	for filter_file,_,_,_ in arglist:
		filt = np.load(temp_filter_path + filter_file + '.npy')
		filters.append(filt)

	filters = np.asarray(filters)
	f_shape = filters.shape
	filters = filters.reshape(f_shape[0]*f_shape[1], f_shape[2])
	kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(filters)
	dictionary = kmeans.cluster_centers_
	np.save('dictionary.npy', dictionary)




