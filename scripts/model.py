# libraries
import numpy as np
import pandas as pd
import os
import cv2
# setting random seed
seed = 77
np.random.seed(seed)
# libraries
from keras.models import Sequential
from keras import optimizers
from keras.layers.convolutional import Cropping2D
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Lambda
import math
import traceback
import pylab

# target path for model
model_save_path = "/full_path/model.h5"

# constants
batch_size = 20.0
learning_rate = 0.0009
epochs = 75
input_shape = (160, 320, 3)
len_train_X = 0
len_val_X = 0

# getting length of train and validation data - used in model.fit_generator
len_train_X = len(pd.read_csv("/full_path/train_data.csv"))
len_val_X = len(pd.read_csv("/full_path/val_data.csv"))
print("length of train data: ", len_train_X)
print("length of validation data: ", len_val_X)

# generator function for training
def train_generator():
	try:
		# global variables
		global batch_size, input_shape, len_train_X
		# paths
		train_csv_path = "/full_path/train_data.csv"
		# reading csv
		data = pd.read_csv(train_csv_path)
		# getting length of data
		len_train_X = len(data)
		# separating image-file-name and steering
		image_files = data['images']
		steering = data['steering']
		# runing generator
		while(True):
			for index in range(int(math.ceil(len_train_X/batch_size))):
				# making batches of size batch_size
				current_image_bucket = list(image_files[index * int(batch_size) : (index + 1) * int(batch_size)])
				current_label_bucket = list(steering[index * int(batch_size) : (index + 1) * int(batch_size)])
				# making numpy array for storing pixel values and labels
				train_X = np.ndarray((len(current_image_bucket), input_shape[0], input_shape[1], input_shape[2]))
				train_Y = np.ndarray((len(current_label_bucket)))
				# reading each image and its label of current batch and storing in their respective numpy array
				for i in range(len(current_image_bucket)):
					train_X[i] = cv2.imread(current_image_bucket[i].strip())
					train_Y[i] = current_label_bucket[i]
				# passing data batch for training
				yield train_X, train_Y
	except Exception as e:
		traceback.print_exc()

# generator function for validation
def val_generator():
	try:
		# global variables
		global batch_size, input_shape, len_val_X
		# paths
		val_csv_path = "/full_path/val_data.csv"
		# reading csv
		data = pd.read_csv(val_csv_path)
		# getting length of validation data
		len_val_X = len(data)
		# separating image-file-name and steering
		image_files = data['images']
		steering = data['steering']
		# runing generator
		while(True):
			for index in range(int(math.ceil(len_val_X/batch_size))):
				# making batch of data
				current_image_bucket = list(image_files[index * int(batch_size) : (index + 1) * int(batch_size)])
				current_label_bucket = list(steering[index * int(batch_size) : (index + 1) * int(batch_size)])
				# making numpy array for storing image pixels and labels
				val_X = np.ndarray((len(current_image_bucket), input_shape[0], input_shape[1], input_shape[2]))
				val_Y = np.ndarray((len(current_label_bucket)))
				# reading each image and its label, saving in its respective numpy array
				for i in range(len(current_image_bucket)):
					val_X[i] = cv2.imread(current_image_bucket[i].strip())
					val_Y[i] = current_label_bucket[i]
				# passing data for validation
				yield val_X, val_Y
	except Exception as e:
		traceback.print_exc()

# getting validation label for validation purpose
val_csv_path = "/full_path/val_data.csv"
val_Y = np.array(pd.read_csv(val_csv_path)['steering'])

# making CNN (Convolutional Neural Network)
try:
	model = Sequential()
	# preprocessing data
	model.add(Lambda(lambda x:x/255.0, input_shape=input_shape))
	model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=input_shape))
	# CNN layers
	model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='valid', activation='selu'))
	model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='valid', activation='selu'))
	model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='selu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dense(128, activation='selu'))
	model.add(Dropout(0.2))
	model.add(Dense(1))
	# CNN optimizer
	optimizer = optimizers.Adadelta(lr=learning_rate)
	# compiling model
	model.compile(loss="mean_squared_error", optimizer=optimizer)
	# training CNN model
	model.fit_generator(train_generator(), epochs=epochs, steps_per_epoch=int(math.ceil(len_train_X/batch_size)), max_queue_size=batch_size, verbose=1)
	# validating trained model
	val_score = model.evaluate_generator(val_generator(), steps=int(math.ceil(len_val_X/batch_size)), max_queue_size=batch_size)
	pred = model.predict_generator(val_generator(), steps=int(math.ceil(len_val_X/batch_size)), max_queue_size=batch_size)
	print("Validation MSE: " + str(val_score))
	# getting predictions for validation data
	pred = np.reshape(pred, [len(pred)])
	# making plot for validation data
	pylab.plot(range(0, len(val_Y)), val_Y, '-r', label='actual')
	pylab.plot(range(0, len(val_Y)), pred, '-b', label='predict')
	# saving trained model
	model.save(model_save_path)
	# showing plot
	pylab.show()
except Exception as e:
	traceback.print_exc()
