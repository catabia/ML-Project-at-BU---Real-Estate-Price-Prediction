from __future__ import print_function
import tensorflow as tf
import keras
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.applications.resnet50 import ResNet50
from keras import backend as K
import os

'''A preprocessing procedure for the images. Images corresponding to listings
are passed through a finetuned custom ResNet-like neural network. The image and its vector are printed together.
'''

image_size = 2048
images_per_listing = 15
path_to_images = '/scratch/mona/download/cs542/temp_folder_for_experiments/validation_sets/'
outfile = 'resnet_finetuned_images.npy'
#image_data = np.zeros(30722)

'''Load the metadata into memory'''
data = np.load('clean2.npy')
data = data[:, :2]

'''Load finetuned ResNet50'''
finetuned_resnet = load_model('/scratch/mona/download/cs542/temp_folder_for_experiments/finetune/checkpoints_luxury3/finetune-08.hdf5')
model = Model(inputs=finetuned_resnet.input, outputs=finetuned_resnet.get_layer('flatten_1').output)

image_data = np.zeros((data.shape[0],30722))
image_data[:,:2] = data[:,:2]

#Preprocess images through finetuned ResNet50
images_current_listing = np.zeros((15,224,224,3))
for i in range(data.shape[0]):
	#Load the 15 images corresponding to the i-th listing to memory
	folder_name = str(int(data[i,1]))+'/'
	image_prefix = str(int(data[i,0]))
	flag = 0 #0 value indicates that all the images for the current listing are readable.
	for j in range(15):
		image_name = path_to_images+folder_name+image_prefix+'_'+str(j)+'.jpg'

		if(os.path.exists(image_name)):
			temp = cv2.imread(image_name)
			images_current_listing[j] = cv2.resize(temp, (224,224))
		else:
			flag = 1

	if flag == 0: #Preprocess only if all images for the listing can be opened.
		preprocessed_images = model.predict(images_current_listing)
		image_data[i,2:] = preprocessed_images.reshape(30720)
		
np.save(outfile, image_data)


