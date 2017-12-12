#from sklearn.model_selection import StratifiedKFold
from __future__ import print_function
import tensorflow as tf
import keras
import numpy as np
import cv2
import csv
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model

from keras.layers import Dense, Flatten, Conv2D
from keras.applications.resnet50 import ResNet50
from keras import backend as K
import os

'''A preprocessing procedure for the images. Images corresponding to listings
are passed through ResNet50. The image and its vector are printed together.
'''

output_file = open('images_preprocessed_through_resnet.txt', 'w')

with open('./images_list.txt') as f:
	image_files = f.readlines() 

image_files = [x.strip() for x in image_files] #Load the names of images into the list image_files

#Using ResNet50 trained on ImageNet 
resnet50 = ResNet50(weights = 'imagenet', input_shape= (224, 224, 3))
model1 = Model(inputs=resnet50.input, outputs=resnet50.get_layer('flatten_1').output)

input_to_resnet = np.zeros((1,224,224,3)) #Initialize an array as the input to resnet

counter = 0
while counter < len(image_files):

	for img_file in image_files[counter:counter+15]:
		file_path = '/scratch2/cs542/foxy_dataset/images/' + img_file
		if(os.path.exists(file_path)):
			temp = cv2.imread(file_path)
			#resize our input images to 224*224 so that we can feed them to ResNet50 for prediction
			temp = cv2.resize(temp, (224,224)) #Dimensions are to be switched is because numpy and cv2 treats height and width in opposite order.
			input_to_resnet[0] = temp
			preprocessed_image = model1.predict(input_to_resnet)
			output_file.write(img_file+str(" : "))
			for k in range(2048):
				output_file.write(str(preprocessed_image[0,k])+str(", "))
			output_file.write("\n")
		else:
			continue
	counter+=15
