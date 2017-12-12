import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import backend as K
import matplotlib.pyplot as plt

epochs = 1000
batch_size = 32

#Defining accuracy metric passing as input for metric in model.compile
def ten_percent_accuracy(y_true, y_pred):
	return K.mean((K.abs(y_pred - y_true)/y_true) <= 0.1)

def three_percent_accuracy(y_true, y_pred):
	return K.mean((K.abs(y_pred - y_true)/y_true) <= 0.03)

def load_data():
	image_data = np.load('../resnet_finetuned_images.npy')
	data = np.load('../clean2.npy')
	return np.hstack((data[:,:3], image_data[:, 2:]))

def split_data(data):
	train_data = data[np.argwhere(data[:,1]<=7),:]
	test_data = data[np.argwhere(data[:,1] == 8)]
	cv_data = data[np.argwhere(data[:,1] == 9)] 
	
	#Reshape the data, as each of the above numpy arrays are 3 dimensional. The middle dimension is not needed and it creates problems with scaling.
	train_data = np.reshape(train_data,(train_data.shape[0],train_data.shape[2]))
	test_data = np.reshape(test_data,(test_data.shape[0],test_data.shape[2]))
	cv_data = np.reshape(cv_data,(cv_data.shape[0],cv_data.shape[2]))
	
	#Split into X part and y part
	
	Xtrain = train_data[:,3:] 
	Ytrain = train_data[:,2]
	Xtest = test_data[:,3:]
	Ytest = test_data[:,2]
	Xcrossvalid = cv_data[:,3:]
	Ycrossvalid = cv_data[:,2]
	
	return Xtrain, Ytrain, Xtest, Ytest, Xcrossvalid, Ycrossvalid


def scale_training_data(X_train): #Scale training data (MinMaxScaling) and save the scaling transform so as to use the same transform later on test data (that is necessary)
	scaler = preprocessing.MinMaxScaler()
	X_train = scaler.fit_transform(X_train)
	return X_train, scaler

np.random.seed(7823)

data = load_data()
Xtrain, Ytrain, Xtest, Ytest, Xcrossvalid, Ycrossvalid = split_data(data)

Xtrain = np.vstack((Xtrain, Xcrossvalid))
Ytrain = np.hstack((Ytrain, Ycrossvalid))

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
model.add(Dense(512, input_dim=Xtrain.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='relu'))

#model.compile(optimizer='rmsprop', loss='mape',metrics=['mae', 'mse', ten_percent_accuracy])
model.compile(optimizer='rmsprop', loss='mse',metrics=['mae', 'mape', ten_percent_accuracy])

#model.compile(optimizer='rmsprop', loss='mse',metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity'])
#If we need to try with Adam optimizer, comment the above line and uncomment the line below this
#model.compile(optimizer='adam', loss='mse',metrics=['mean_squared_error'])

#Object to control early stopping

#early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
early_stopping = EarlyStopping(monitor='val_mean_absolute_percentage_error', min_delta=0, patience=5, verbose=0, mode='auto')

training_history = model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=epochs, validation_split = 0.1, shuffle=True, callbacks=[early_stopping])

#Plotting the training graph

plt.plot(training_history.history['loss'])
plt.plot(training_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(training_history.history['mean_absolute_percentage_error'])
plt.plot(training_history.history['val_mean_absolute_percentage_error'])
plt.title('Model Accuracy')
plt.ylabel('MAPE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

score = model.evaluate(Xtest, Ytest, batch_size=batch_size)

Ypred = model.predict(Xtest)
print(np.column_stack((Ytest, Ypred))[:100])

print("Test loss is ",score)
#model.save('metadata_only_fully_trained_keras_model_21.h5')
