import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words

import skimage.io
from multiprocessing import Pool
from sklearn.metrics import confusion_matrix


def build_recognition_system(num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,M)
	* labels: numpy.ndarray of shape (N)
	* dictionary: numpy.ndarray of shape (K,3F)
	* SPM_layer_num: number of spatial pyramid layers
	'''

	K = 150
	layer_num = 3
	train_data = np.load("../data/train_data.npz")
	dictionary = np.load("dictionary.npy")

	labels = []
	arglist = []
	for img_details, label in zip(train_data['image_names'], train_data['labels']):
		img_path = str('../data/' + img_details[0])
		args = (img_path, dictionary, layer_num, K)
		arglist.append(args)
		labels.append(label)

	p = Pool(num_workers)
	features = p.starmap(get_image_feature, arglist)
	features = np.asarray(features)
	labels = np.asarray(labels)
	np.savez('trained_system.npz',dictionary=dictionary,features=features,labels=labels,SPM_layer_num=layer_num)




def evaluate_recognition_system(num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''


	test_data = np.load("../data/test_data.npz")
	trained_system = np.load("trained_system.npz")
	histograms = trained_system['features']
	dictionary = trained_system['dictionary']
	layer_num = trained_system['SPM_layer_num']
	train_labels = trained_system['labels']

	K = 150
	arglist = []
	test_labels = []
	for img_details, test_label in zip(test_data['image_names'], test_data['labels']):
		img_path = str('../data/' + img_details[0])
		args = (img_path, dictionary, layer_num, K)
		arglist.append(args)
		test_labels.append(test_label)

	p = Pool(num_workers)
	test_features = p.starmap(get_image_feature, arglist)
	test_features = np.asarray(test_features)
	test_labels = np.asarray(test_labels)

	predicted_labels = []
	for word_hist in test_features:
		distance_metric = distance_to_set(word_hist, histograms)
		train_img_idx = np.argmax(distance_metric)
		predicted_labels.append(train_labels[train_img_idx])

	predicted_labels = np.asarray(predicted_labels)
	C = confusion_matrix(test_labels, predicted_labels)
	accuracy = (np.trace(C) * 1.0)/ C.sum()
	return C, accuracy





def get_image_feature(file_path,dictionary,layer_num,K):
	'''
	Extracts the spatial pyramid matching feature.

	[input]
	* file_path: path of image file to read
	* dictionary: numpy.ndarray of shape (K,3F)
	* layer_num: number of spatial pyramid layers
	* K: number of clusters for the word maps

	[output]
	* feature: numpy.ndarray of shape (K)
	'''
	# print ('getting_features...')
	image = skimage.io.imread(file_path)
	image = image.astype('float')/255
	wordmap = visual_words.get_visual_words(image, dictionary)
	features = get_feature_from_wordmap_SPM(wordmap, layer_num, K)
	return features



def distance_to_set(word_hist,histograms):
	'''
	Compute similarity between a histogram of visual words with all training image histograms.

	[input]
	* word_hist: numpy.ndarray of shape (K)
	* histograms: numpy.ndarray of shape (N,K)

	[output]
	* sim: numpy.ndarray of shape (N)
	'''
	return np.minimum(word_hist, histograms).sum(axis=-1)



def get_feature_from_wordmap(wordmap,dict_size):
	'''
	Compute histogram of visual words.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* dict_size: dictionary size K

	[output]
	* hist: numpy.ndarray of shape (K)
	'''

	hist = np.histogram(wordmap.flatten(), bins=dict_size, range=(0,dict_size-1))[0]
	hist_norm = np.linalg.norm(hist, ord=1)
	hist = hist / hist_norm
	return hist



def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
	def blockshaped(arr, nrows, ncols):
	    """
	    Return an array of shape (n, nrows, ncols) where
	    n * nrows * ncols = arr.size

	    If arr is a 2D array, the returned array should look like n subblocks with
	    each subblock preserving the "physical" layout of arr.
	    """
	    h, w = arr.shape
	    return (arr.reshape(h//nrows, nrows, -1, ncols)
	               .swapaxes(1,2)
	               .reshape(-1, nrows, ncols))
	'''
	Compute histogram of visual words using spatial pyramid matching.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* layer_num: number of spatial pyramid layers
	* dict_size: dictionary size K

	[output]
	* hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
	'''
	nearest_multiple_row = wordmap.shape[0] - (wordmap.shape[0] % 4)
	nearest_multiple_col = wordmap.shape[1] - (wordmap.shape[1] % 4)
	wordmap = wordmap[:nearest_multiple_row, :nearest_multiple_col]
	weights = [1.0/4, 1.0/4, 1.0/2]
	hist_all = []
	for l in range(layer_num):
		cell_count = 2**l * 1.0
		weight = weights[l]
		nrows, ncols = wordmap.shape
		nrows, ncols = int(nrows/cell_count), int(ncols/cell_count)
		for block in blockshaped(wordmap, nrows, ncols):
			hist = get_feature_from_wordmap(block,dict_size)
			hist_all.append(hist * weight)

	hist_all = np.asarray(hist_all)
	hist_all = hist_all.flatten()
	hist_all = hist_all / np.linalg.norm(hist_all, ord=1)
	return hist_all








	

