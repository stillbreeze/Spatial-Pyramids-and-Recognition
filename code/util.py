import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import multiprocessing
import itertools

def get_num_CPU():
	'''
	Counts the number of CPUs available in the machine.
	'''
	return multiprocessing.cpu_count()

def get_VGG16_weights():
	'''
	Returns the weights and meta-information of the VGG-16 network.

	[output]
	* vgg16_weights: numpy.ndarray of shape (L,3)
					 The first column stores the type of network layer.
					 If the type is "conv2d" or "linear", the second and third column stores the weight and bias.
					 If the type is "maxpool2d", the second column stores the kernel size (assuming the same stride size).
	'''

	vgg16 = torchvision.models.vgg16(pretrained=True)
	vgg16_weights = []

	for layer in vgg16.features:
		if isinstance(layer,torch.nn.Conv2d):
			l = ["conv2d",layer.weight.detach().numpy(),layer.bias.detach().numpy()]
		if isinstance(layer,torch.nn.ReLU):
			l = ["relu"]
		if isinstance(layer,torch.nn.MaxPool2d):
			l = ["maxpool2d",layer.kernel_size]
		vgg16_weights.append(l)
	for layer in vgg16.classifier:
		if isinstance(layer,torch.nn.Linear):
			l = ["linear",layer.weight.detach().numpy(),layer.bias.detach().numpy()]
		if isinstance(layer,torch.nn.ReLU):
			l = ["relu"]
		if isinstance(layer,torch.nn.Dropout): continue
		vgg16_weights.append(l)

	return vgg16_weights

def display_filter_responses(response_maps):
	'''
	Visualizes the filter response maps.

	[input]
	* response_maps: a numpy.ndarray of shape (H,W,3F)
	'''
	
	fig = plt.figure(1)
	
	j = 1
	for i in range(20):
		plt.subplot(4,5,i+1)
		resp = response_maps[:,:,i*3:i*3+3]
		resp_min = resp.min(axis=(0,1),keepdims=True)
		resp_max = resp.max(axis=(0,1),keepdims=True)
		resp = (resp-resp_min)/(resp_max-resp_min)
		plt.imshow(resp)
		plt.axis("off")

	plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05,wspace=0.05,hspace=0.05)
	plt.show()

def save_wordmap(wordmap, filename):
	fig = plt.figure(2)
	plt.axis('equal')
	plt.axis('off')
	plt.imshow(wordmap, cmap=plt.get_cmap('rainbow'))
	plt.savefig(filename, pad_inches=0)


def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

	plt.figure()
	plt.show()
