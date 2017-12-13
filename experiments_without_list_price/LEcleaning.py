#Lightning Export dataset cleaning

import numpy as np

#import the data, and make sure the formatting is correct
# PLEASE NOTE: the dtype of this array is float64
# that is to say, each entry is of type float64
csv = np.genfromtxt('LightningExport2.csv', delimiter = ",", usecols=np.arange(0,21), skip_header=1)

#extract the 9 useful columns
csv = csv[:, [0, 2, 3, 12, 14, 15, 16, 17, 18]]
indices = np.argsort = [0,2,1,4,5,6,7,8,3]
csv = csv[:, indices]
#the columns of the csv now represent
# 0 - MLS number
# 1 - soldprice (our y)
# 2 - listprice
# 3 - bedroom count
# 4 - bathroom count
# 5 - sqft of property
# 6 - age
# 7 - lot size

#elimnating rows with nan
#originally, we have 54760 listings
#after we remove the rows with NaNs, we have 54696 listings
#this preserves over 99.9% of listings, so it seems like a reasonably sane action to take
csv = csv[~np.isnan(csv).any(axis=1)]

#remove duplicates
#now we have 53,167 listings!
unique_keys, indices = np.unique(csv[:,0], return_index=True)
csv = csv[indices]

#change zip code to categorical - 1-hot
#we appear to have 621 unique zips!
zips = csv[:,8]
unique_zips = np.sort(np.unique(zips))
num_zips = np.unique(zips).shape[0]

onehot = np.zeros((zips.shape[0], num_zips))
for i in range(zips.shape[0]):
	index = np.where(unique_zips == zips[i])
	onehot[i,index] = 1

np.save("nothot", csv)

#replace last column in the array with 621 columns of one-hot encoding
csv = np.delete(csv, 8, axis = 1)
csv = np.concatenate((csv, onehot), axis=1)




#save the clean data
#np.save("clean", csv)

# counter = 1
# for row in csv:
# 	print(counter)
# 	print(row)
# 	counter +=1