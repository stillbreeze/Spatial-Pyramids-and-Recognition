import numpy as np
import multiprocessing
import threading
import queue
import imageio
import os,time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers

import skimage.io
import skimage.transform
from multiprocessing import Pool
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist

def build_recognition_system(vgg16,num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,K)
	* labels: numpy.ndarray of shape (N)
	'''
	new_classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-2])
	vgg16.classifier = new_classifier
	vgg16.eval()

	train_data = np.load("../data/train_data.npz")

	labels = []
	features = []
	for img_details, label in zip(train_data['image_names'], train_data['labels']):
		img_path = str('../data/' + img_details[0])
		feature = get_image_feature([img_path, vgg16])
		features.append(feature)
		labels.append(label)
	features = np.asarray(features)
	labels = np.asarray(labels)
	np.savez('trained_system_deep.npz',features=features,labels=labels)



def evaluate_recognition_system(vgg16,num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''

	new_classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-2])
	vgg16.classifier = new_classifier
	vgg16.eval()
	
	test_data = np.load("../data/test_data.npz")
	trained_system = np.load("trained_system_deep.npz")
	train_features = trained_system['features']
	train_labels = trained_system['labels']

	test_labels = []
	test_features = []
	for img_details, label in zip(test_data['image_names'], test_data['labels']):
		img_path = str('../data/' + img_details[0])
		feature = get_image_feature([img_path, vgg16])
		test_features.append(feature)
		test_labels.append(label)
	test_features = np.asarray(test_features)
	test_labels = np.asarray(test_labels)

	predicted_labels = []
	for feature in test_features:
		distance_metric = distance_to_set(feature, train_features)
		train_img_idx = np.argmax(distance_metric)
		predicted_labels.append(train_labels[train_img_idx])

	predicted_labels = np.asarray(predicted_labels)
	C = confusion_matrix(test_labels, predicted_labels)
	accuracy = (np.trace(C) * 1.0)/ C.sum()
	return C, accuracy


def preprocess_image(image):
	'''
	Preprocesses the image to load into the prebuilt network.

	[input]
	* image: numpy.ndarray of shape (H,W,3)

	[output]
	* image_processed: torch.Tensor of shape (3,H,W)
	'''
	if image.ndim == 1:
		image = np.stack((image,)*3, -1)
	elif image.shape[-1] == 4:
		image = image[:,:,:3]
	mean = np.array([0.485,0.456,0.406])
	std = np.array([0.229,0.224,0.225])
	image = skimage.transform.resize(image, (224,224))
	image = (image - mean)/std
	image = np.moveaxis(image, -1,0)
	image = torch.from_numpy(image)
	image = image.unsqueeze(0)
	return image


def get_image_feature(args):
	'''
	Extracts deep features from the prebuilt VGG-16 network.
	This is a function run by a subprocess.
 	[input]
	* i: index of training image
	* image_path: path of image file
	* vgg16: prebuilt VGG-16 network.
	* time_start: time stamp of start time
 	[saved]
	* feat: evaluated deep feature
	'''
	image_path,vgg16 = args
	# print ('getting_features...')
	image = skimage.io.imread(image_path)
	image = preprocess_image(image)
	fc7_out = vgg16(image)
	return fc7_out.detach().numpy()[-1]



def distance_to_set(feature,train_features):
	'''
	Compute distance between a deep feature with all training image deep features.

	[input]
	* feature: numpy.ndarray of shape (K)
	* train_features: numpy.ndarray of shape (N,K)

	[output]
	* dist: numpy.ndarray of shape (N)
	'''
	return -cdist(np.expand_dims(feature,0), train_features, 'euclidean')




