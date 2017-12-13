# luxtune.py

#adapted from:  https://stackoverflow.com/questions/43867032/how-to-fine-tune-resnet50-in-keras
#and also from: https://machinelearningmastery.com/check-point-deep-learning-models-keras/
#and from: https://fizzylogic.nl/2017/05/08/monitor-progress-of-your-keras-based-neural-network-using-tensorboard/
#also from: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/ 

import tensorflow as tf
import keras
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import h5py


# define the training generator
# # train set:  Found 149994 images belonging to 7 classes.
train_datagen = ImageDataGenerator(fill_mode = 'constant',
				horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
     		"/scratch2/cs542/luxury/Train",
    		target_size=(224, 224),
    		batch_size=75,
    		class_mode='categorical')

# define test/training generator
# test set:  Found 8545 images belonging to 7 classes.
valid_datagen = ImageDataGenerator(fill_mode = 'constant')

valid_generator = valid_datagen.flow_from_directory(
     		"/scratch2/cs542/luxury/Test",
    		target_size=(224, 224),
    		batch_size=75,
    		class_mode='categorical')


# create the base pre-trained model
resnet50 = ResNet50 (weights = 'imagenet', input_shape= (224, 224, 3), include_top=True)
base_model = Model(inputs=resnet50.input, outputs=resnet50.get_layer('flatten_1').output)

# add a logistic layer - 7 classes
x = base_model.output
predictions = Dense(7, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the bottom layers
# freeze up through the activation of res2 (res2c_branch2c, add3, activation_10)
# total 176 layers in this model
for layer in model.layers[:37]:
   layer.trainable = False
for layer in model.layers[37:]:
   layer.trainable = True

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics = ['accuracy'])

# model checkpoints
filepath="/scratch/mona/download/cs542/temp_folder_for_experiments/finetune/checkpoints_luxury2/finetune-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, save_best_only=False)

# train the model on the new data for a few epochs
history = model.fit_generator(
        	train_generator,
		steps_per_epoch = 500,
        	epochs=25,
        	validation_data=valid_generator,
		validation_steps = 25,
		callbacks = [checkpoint])
#1999, 113

#print(model.summary())
print(history.history)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('graphs_luxury/acc_lux.pdf')
#plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('graphs_luxury2/loss_lux.pdf')
#plt.show()
