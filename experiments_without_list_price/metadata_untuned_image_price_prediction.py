import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
from keras import backend as K
import matplotlib.pyplot as plt

epochs = 100
batch_size = 32

def ten_percent_accuracy(y_true, y_pred):
        return K.mean((K.abs(y_pred - y_true)/y_true) <= 0.1)

def load_data():
	data = np.load('../clean2.npy')
	data = data[:-1, :]
	image_data = np.load('../resnet_not_finetuned_images.npy')
	return np.hstack((data, image_data[:, 2:]))
	

def split_data(data):
	train_data = data[np.argwhere(data[:,1]<=7),:]
	test_data = data[np.argwhere(data[:,1] == 8)]
	cv_data = data[np.argwhere(data[:,1] == 9)] 
	
	#Reshape the data, as each of the above numpy arrays are 3 dimensional. The middle dimension is not needed and it creates problems with scaling.
	train_data = np.reshape(train_data,(train_data.shape[0],train_data.shape[2]))
	test_data = np.reshape(test_data,(test_data.shape[0],test_data.shape[2]))
	cv_data = np.reshape(cv_data,(cv_data.shape[0],cv_data.shape[2]))
	
	#Split into X part and y part
	
	#List price is very close to sold price in most cases
	#Xtrain = train_data[:,3:] # including list price
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

np.random.seed(7823)

data = load_data()
Xtrain, Ytrain, Xtest, Ytest, Xcrossvalid, Ycrossvalid = split_data(data)

Xtrain = np.vstack((Xtrain, Xcrossvalid))
Ytrain = np.hstack((Ytrain, Ycrossvalid))

#Xtrain = Xtrain[:1000,:]
#Ytrain = Ytrain[:1000]

#Xtest = Xtest[:100,:]
#Ytest = Ytest[:100]

'''
X_train_noncategorical_scaled, scaling_transform = scale_training_data(Xtrain[:,0:6]) #Pass the noncategorical data to the scaling subroutine
Xtrain[:,0:6] = X_train_noncategorical_scaled

#Use the scaling model returned by scale_training_data() to transform test data (only the noncategorical part).
Xtest[:,0:6] = scaling_transform.fit_transform(Xtest[:,0:6])
'''

#Setting training size
train_size = Xtrain.shape[0]
test_size = Xtest.shape[0]
num_batches = int(train_size/batch_size)

#A multi input neural network model to pass images and metadat separately
image_input = Input(shape=(30720,))
#image_hidden = Dense(512, activation='relu')(image_input)
#Uncomment the following line (or make copies of the following lines to create more layers
#image_hidden = Dense(512, activation='relu')(image_hidden)
#image_output = Dense(10, activation='relu')(image_hidden)
image_output = Dense(512, activation='relu')(image_input)

metadata_input = Input(shape=(Xtrain.shape[1]-30720,)) #Fill in the input shape
#metadata_hidden = Dense(512, activation='relu')(metadata_input)
#Uncomment the following line (or make copies of the following lines to create more layers
#metadata_hidden = Dense(512, activation='relu')(metadata_hidden)
#metadata_output = Dense(10, activation='relu')(metadata_hidden)

#complete_data = keras.layers.concatenate([metadata_output, image_output])
complete_data = keras.layers.concatenate([metadata_input, image_output])
complete_hidden = Dense(64, activation='relu')(complete_data)
#complete_hidden = Dense(128, activation='relu')(complete_hidden)
complete_output = Dense(1, activation='relu')(complete_hidden)


model = Model(inputs=[metadata_input, image_input], outputs=complete_output)

model.compile(optimizer='rmsprop', loss='mse',metrics=['mae', 'mape', ten_percent_accuracy])
#model.compile(optimizer='rmsprop', loss='mape',metrics=['mae', 'mse', ten_percent_accuracy])

#model.compile(optimizer='rmsprop', loss='mse',metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity'])
#If we need to try with Adam optimizer, comment the above line and uncomment the line below this
#model.compile(optimizer='adam', loss='mse',metrics=['mean_squared_error'])

#Object to control early stopping

#early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
early_stopping = EarlyStopping(monitor='val_mean_absolute_percentage_error', min_delta=0, patience=5, verbose=0, mode='auto')

training_history = model.fit([Xtrain[:,:-30720], Xtrain[:,-30720:]], Ytrain, batch_size=batch_size, epochs=epochs, validation_split = 0.1, shuffle=True, callbacks=[early_stopping])

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

score = model.evaluate([Xtest[:,:-30720], Xtest[:,-30720:]], Ytest, batch_size=batch_size)

Ypred = model.predict([Xtest[:,:-30720], Xtest[:,-30720:]])
print(np.column_stack((Ytest, Ypred))[:100])

print("Test loss is ",score)

#model.save('metadata_untuned_images_fully_trained_model_1.h5')
