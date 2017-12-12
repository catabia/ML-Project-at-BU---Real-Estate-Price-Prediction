#from sklearn.model_selection import StratifiedKFold
from __future__ import print_function
import tensorflow as tf
import keras
import numpy as np
import cv2
import csv
import re
#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model

from keras.layers import Dense, Flatten
#from keras.layers.convolutional import Convolution2D
#from keras.applications.resnet50 import ResNet50
from keras import backend as K
import os
from itertools import islice

epochs = 100
batch_size = 32
train_size = 2000
test_size = 250
image_size = 2048
images_per_listing = 15
num_batches = int(train_size/batch_size)

pattern = re.compile(r"(\d*)(_\d*.jpg)") #Image file name pattern

'''Load the data into memory'''
def load_data():
	x_train = np.zeros((train_size, image_size * images_per_listing))
	y_train = np.zeros(train_size)
	
	x_test = np.zeros((test_size, image_size * images_per_listing))
	y_test = np.zeros(test_size)
	
	csv_file_path = 'tmks_filtered.csv'
	processed_image_file = 'images_preprocessed_through_resnet.txt'
	
	data = np.zeros((1,2)) # CSV file data
	with open(csv_file_path, 'r', encoding ='utf-8') as f:
		reader = csv.DictReader(f, delimiter=',')
		for row in reader:
			temp = np.array([row['MLS#'],row['LIST PRICE'].replace("-","")])
			temp = np.reshape(temp, (1, 2))
			data = np.concatenate((data,temp), axis=0)

	data = np.delete(data, 0, axis=0)
	'''
	print('Printing a clash')
	print(data[35935])
	print(data[35936])
	print('end')
	'''
	#Load the first train_size images into memory
	counter = 0
	current_listing = None
	current_listing_data = []
	with open(processed_image_file) as myfile:
		for line in islice(myfile, 1 + (train_size + test_size) * images_per_listing):
			#Read a line from the file and convert it to a list.
			x_datapoint = line.strip()
			x_datapoint = x_datapoint.replace(" ","")
			x_datapoint = x_datapoint.split(",")
			temp = x_datapoint[0].split(":")
			x_datapoint[0] = temp[1]
			
			#Find the corresponding listing in the CSV file and store its price in y_train
			match = re.search(pattern, str(temp[0]))
			if match is not None:
				listing_number = str(match.group(1))
				#print(listing_number)
				temp = [float(listitem) for listitem in x_datapoint[:2048]]

				if(current_listing == listing_number):
					current_listing_data.extend(temp)

				elif(current_listing is not None):
					listing_csv_entry = np.amax(np.argwhere(data[:,0]==current_listing))
					current_listing_data.extend([0] * ((image_size * images_per_listing) - len(current_listing_data)))
					if (counter < train_size):
						#print(x_datapoint)
						x_train[counter] = current_listing_data
						#print(x_train[counter], len(x_train[counter]))
						#print(data[listing_csv_entry,1], ' ', listing_csv_entry)
						#print(data.shape)
						#print(data[listing_csv_entry,1].shape)
						y_train[counter] = data[listing_csv_entry,1].astype(float)
					else:
						x_test[counter-train_size] = current_listing_data
						y_test[counter-train_size] = data[listing_csv_entry,1].astype(float)
					counter += 1
					current_listing = listing_number
					current_listing_data = temp

				else:
					current_listing = listing_number
					current_listing_data = temp

			if(counter >= train_size + test_size):
				break

	return (x_train, y_train), (x_test, y_test)

#Main program 
(x_train, y_train), (x_test, y_test) = load_data()

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

print(x_train.shape)
print(x_test.shape)

#Neural network description
model = Sequential()
model.add(Dense(600, input_dim=x_train.shape[1], activation='relu'))
#model.add(Dense(8000, activation='relu'))
#model.add(Dense(2000, activation='relu'))
#model.add(Dense(1000, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(optimizer='rmsprop', loss='mse',metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split = 0.2, shuffle=True)

score = model.evaluate(x_test, y_test, batch_size=batch_size)

y_predicted = model.predict(x_test)
print(np.column_stack((y_test, y_predicted)))
print("Test loss is ",score)
'''			
######### Generate a list of random numbers
rand_array = np.random.choice(lines_in_file, train_size, replace = False)
rand_array = np.sort(rand_array)

#Load CSV data into memory

csv_file_path = 'tmks_filtered.csv'


#Process in batches. Create x_train and y_train




fp = open("images_preprocessed_through_resnet.txt")
file_pos = 0
counter = 0
for i in range(num_batches):
	### Load X data into memory
	batch_set = rand_array[counter:counter+batch_size]
	x_train = np.zeros((batch_size, 2048))
	y_train = np.zeros(batch_size)
	
	for j in batch_set:
		while file_pos < j:
			fp.readline()
			file_pos += 1
		line = fp.readline()
		file_pos += 1
		line = line.strip()
		line = line.split(",")
		
'''
