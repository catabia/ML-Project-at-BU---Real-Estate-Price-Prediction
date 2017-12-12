from __future__ import print_function
import numpy as np
import csv
import re

import os
from itertools import islice

image_size = 2048
images_per_listing = 15

pattern = re.compile(r"(\d*)(_\d*.jpg)") #Image file name pattern

'''Load the data into memory'''
data = np.load('clean2.npy')
data = data[:, :2]

print(data.shape)

#Converting data to dictionary
dict = {}

for i in range(data.shape[0]):
	dict[data[i, 0]] = data[i, 1]

#print(dict)

image_data = np.zeros(30722)
processed_image_file = 'images_preprocessed_through_resnet.txt'
outfile = 'resnet_processed_images_2.npy'
current_listing = None
current_listing_data = []

with open(processed_image_file) as myfile:
	for line in myfile:
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
			temp = [float(listitem) for listitem in x_datapoint[:2048]]
			if(current_listing == listing_number):
				current_listing_data.extend(temp)

			elif(current_listing is not None):
				if float(current_listing) in dict:
					current_listing_data.extend([0] * ((image_size * images_per_listing) - len(current_listing_data)))
					listing_data = [float(current_listing), dict[float(current_listing)]]
					listing_data.extend(current_listing_data)
					image_data = np.vstack((image_data, listing_data))

					#Printing progress
					if(image_data.shape[0] % 100 == 0):
						print(image_data.shape)

				current_listing = listing_number
				current_listing_data = temp

			else:
				current_listing = listing_number
				current_listing_data = temp


image_data = np.delete(image_data, 0, axis = 0)

np.save(outfile, image_data)
print(image_data.shape)
