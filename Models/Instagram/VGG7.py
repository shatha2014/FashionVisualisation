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

class VGG7_model:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# inputShape with theano backend
		model = Sequential()
		inputShape = (depth, width, height)

		# First block
		model.add(Convolution2D(64, 3, 3, border_mode="same", input_shape=inputShape, name= 'conv1-1'))
		model.add(Activation("relu"))
		model.add(Convolution2D(64, 3, 3, border_mode="same", name= 'conv1-2'))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# Second block
		model.add(Convolution2D(128, 3, 3, border_mode="same", name= 'conv2-1'))
		model.add(Activation("relu"))
		model.add(Convolution2D(128, 3, 3, border_mode="same", name= 'conv2-2'))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# Third block
		model.add(Convolution2D(256, 3, 3, border_mode="same", name= 'conv3-1'))
		model.add(Activation("relu"))
		model.add(Convolution2D(256, 3, 3, border_mode="same", name= 'conv3-2'))
		model.add(Activation("relu"))
		model.add(Convolution2D(256, 3, 3, border_mode="same", name= 'conv3-3'))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# FC
		model.add(Flatten())
		model.add(Dense(1024))
		model.add(Activation("relu"))
		model.add(Dropout(0.5))


		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		model.add(Dropout(0.5))


		# return the constructed network architecture
		return model