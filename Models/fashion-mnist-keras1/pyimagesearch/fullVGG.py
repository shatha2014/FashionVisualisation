# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class fullVGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# inputShape with theano backend
		model = Sequential()
		inputShape = (depth, height, width)
		chanDim = 1

		# First block
		model.add(Convolution2D(64, 3, 3, border_mode="same",
						 input_shape=inputShape))
		model.add(Activation("relu"))
		#model.add(BatchNormalization(axis=chanDim))
		model.add(Convolution2D(64, 3, 3, border_mode="same"))
		model.add(Activation("relu"))
		#model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		#model.add(Dropout(0.25))

		# Second block
		model.add(Convolution2D(128, 3, 3, border_mode="same",
						 input_shape=inputShape))
		model.add(Activation("relu"))
		#model.add(BatchNormalization(axis=chanDim))
		model.add(Convolution2D(128, 3, 3, border_mode="same"))
		model.add(Activation("relu"))
		#model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		#model.add(Dropout(0.25))

		# Third block
		model.add(Convolution2D(256, 3, 3,border_mode="same",
						 input_shape=inputShape))
		model.add(Activation("relu"))
		#model.add(BatchNormalization(axis=chanDim))
		model.add(Convolution2D(256, 3, 3, border_mode="same"))
		model.add(Activation("relu"))
		#model.add(BatchNormalization(axis=chanDim))
		model.add(Convolution2D(256, 3, 3, border_mode="same"))
		model.add(Activation("relu"))
		#model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		#model.add(Dropout(0.25))

		# 4th block
		model.add(Convolution2D(512, 3, 3, border_mode="same",
						 input_shape=inputShape))
		model.add(Activation("relu"))
		#model.add(BatchNormalization(axis=chanDim))
		model.add(Convolution2D(512, 3, 3, border_mode="same"))
		model.add(Activation("relu"))
		#model.add(BatchNormalization(axis=chanDim))
		model.add(Convolution2D(512, 3, 3, border_mode="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		#model.add(Dropout(0.25))


		# FC
		model.add(Flatten())
		model.add(Dense(4096))
		model.add(Activation("relu"))
		#model.add(BatchNormalization())
		#model.add(Dropout(0.5))

		model.add(Dense(4096))
		model.add(Activation("relu"))
		#model.add(BatchNormalization())
		#model.add(Dropout(0.5))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model
