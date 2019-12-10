import numpy as np
import scipy.ndimage
import os,time

from scipy.ndimage import convolve as conv2d
import skimage.measure

def extract_deep_feature(x,vgg16_weights):
	'''
	Extracts deep features from the given VGG-16 weights.

	[input]
	* x: numpy.ndarray of shape (H,W,3)
	* vgg16_weights: numpy.ndarray of shape (L,3)

	[output]
	* feat: numpy.ndarray of shape (K)
	'''
	# for layer in vgg16_weights[:31]:
	fc_7_idx = -2
	for layer in vgg16_weights[:fc_7_idx]:
		layer_type = layer[0]
		# print (layer_type)
		# print (x.shape)
		if layer_type == 'conv2d':
			w = layer[1]
			b = layer[2]
			x = multichannel_conv2d(x,w,b)
		elif layer_type == 'maxpool2d':
			size = layer[1]
			x = max_pool2d(x,size)
		elif layer_type == 'relu':
			x = relu(x)
		elif layer_type == 'linear':
			w = layer[1]
			b = layer[2]
			x = linear(x,w,b)
		# print (x.shape)
		# print ('-'*10)
	x = relu(x)
	return x


def multichannel_conv2d(x,weight,bias):
	'''
	Performs multi-channel 2D convolution.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim,kernel_size,kernel_size)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* feat: numpy.ndarray of shape (H,W,output_dim)
	'''
	# print (x.shape)
	# print (weight.shape)
	weight = weight[:,:,::-1,::-1]
	output_dim = weight.shape[0] 
	input_dim = weight.shape[1]
	res = []
	for j in range(output_dim):
		res_k = []
		for k in range(input_dim):
			res_k.append(conv2d(x[:,:,k], weight[j,k,:,:], mode='constant'))
		res_k = np.asarray(res_k).sum(axis=0)
		res.append(res_k)
	res = np.asarray(res)
	res = np.moveaxis(res,0,-1) + bias
	# print (res.shape)
	# print ('*'*10)
	return res


def relu(x):
	'''
	Rectified linear unit.

	[input]
	* x: numpy.ndarray

	[output]
	* y: numpy.ndarray
	'''
	return np.maximum(x,0)

def max_pool2d(x,size):
	'''
	2D max pooling operation.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* size: pooling receptive field

	[output]
	* y: numpy.ndarray of shape (H/size,W/size,input_dim)
	'''
	input_dim = x.shape[-1]
	res = []
	for k in range(input_dim):
		res.append(skimage.measure.block_reduce(x[:,:,k], (size,size), np.max))
	res = np.asarray(res)
	res = np.moveaxis(res,0,-1)
	return res


def linear(x,W,b):
	'''
	Fully-connected layer.

	[input]
	* x: numpy.ndarray of shape (input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* y: numpy.ndarray of shape (output_dim)
	'''
	if x.ndim > 1:
		x = np.moveaxis(x,-1,0)
		return np.matmul(W,x.flatten()) + b
	else:
		return np.matmul(W,x) + b

