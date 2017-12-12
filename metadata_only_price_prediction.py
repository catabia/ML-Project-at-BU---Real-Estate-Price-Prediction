import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import backend as K

epochs = 1000
batch_size = 32

def load_data():
	data = np.load('clean2.npy')
	return data
	

def split_data(data):
	train_data = data[np.argwhere(data[:,1]<=7),:]
	test_data = data[np.argwhere(data[:,1] == 8)]
	cv_data = data[np.argwhere(data[:,1] == 9)] 
	
	#Reshape the data, as each of the above numpy arrays are 3 dimensional. The middle dimension is not needed and it creates problems with scaling.
	train_data = np.reshape(train_data,(train_data.shape[0],train_data.shape[2]))
	test_data = np.reshape(test_data,(test_data.shape[0],test_data.shape[2]))
	cv_data = np.reshape(cv_data,(cv_data.shape[0],cv_data.shape[2]))
	
	#Split into X part and y part
	
	#The commented out lines below use list price as input. That is a problem, as list price is very close to sold price in most cases.
	#Xtrain = train_data[:,3:] 
	Xtrain = train_data[:,4:]
	Ytrain = train_data[:,2]
	#Xtest = test_data[:,3:]
	Xtest = test_data[:,4:]
	Ytest = test_data[:,2]
	#Xcrossvalid = cv_data[:,3:]
	Xcrossvalid = cv_data[:,4:]
	Ycrossvalid = cv_data[:,2]
	
	return Xtrain, Ytrain, Xtest, Ytest, Xcrossvalid, Ycrossvalid


def scale_training_data(X_train): #Scale training data (MinMaxScaling) and save the scaling transform so as to use the same transform later on test data (that is necessary)
	scaler = preprocessing.MinMaxScaler()
	X_train = scaler.fit_transform(X_train)
	return X_train, scaler


data = load_data()
Xtrain, Ytrain, Xtest, Ytest, Xcrossvalid, Ycrossvalid = split_data(data)

'''
X_train_noncategorical_scaled, scaling_transform = scale_training_data(Xtrain[:,0:6]) #Pass the noncategorical data to the scaling subroutine
Xtrain[:,0:6] = X_train_noncategorical_scaled

#Use the scaling model returned by scale_training_data() to transform test data (only the noncategorical part).
Xtest[:,0:6] = scaling_transform.fit_transform(Xtest[:,0:6])
'''

'''Setting training size'''
train_size = Xtrain.shape[0]
test_size = Xtest.shape[0]
num_batches = int(train_size/batch_size)

'''A simple one-hidden-layer neural network model'''
model = Sequential()
model.add(Dense(50, input_dim=Xtrain.shape[1], activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(optimizer='rmsprop', loss='mse',metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

#model.compile(optimizer='rmsprop', loss='mse',metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity'])
#If we need to try with Adam optimizer, comment the above line and uncomment the line below this
#model.compile(optimizer='adam', loss='mse',metrics=['mean_squared_error'])

#Object to control early stopping

#early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
#stopping the training once the loss starts to increase or validation accuracy starts to decrease
#here we monitor the validation loss
#we set min_delta=0 because we are interested in when the loss becomes worse
#patience represents number of epochs before stopping once the loss starts to increase. We leave it at 5
#if the batch size is very small we will see a zig zag pattern so better to use a larger patience value
#however, if we have larger batch size and smaller learning rate, our loss function will be smooth 
#and we could use a smaller patience value.
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto') #changing patience to 5

model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=epochs, validation_split = 0.2, shuffle=True, callbacks=[early_stopping])

score = model.evaluate(Xtest, Ytest, batch_size=batch_size)

print("Test loss is ", score)
model.save('metadata_only_fully_trained_keras_model_3.h5')
