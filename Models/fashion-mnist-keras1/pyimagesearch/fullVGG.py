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
from keras.layers import Input
from keras.models import Model
from keras import applications

class fullVGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		img_input = Input(shape=(depth, width, height))

		model_vgg = applications.VGG16(include_top=False, weights=None, input_tensor=img_input)
		model_vgg.summary()
		#Classification block
		output_vgg16_conv = model_vgg(img_input)
		x = Flatten(name='flatten')(output_vgg16_conv)
		x = Dense(4096, activation='relu', name='fc1')(x)
		x = Dense(4096, activation='relu', name='fc2')(x)
		x = Dense(classes, activation='softmax', name='predictions')(x)

		model = Model(input=img_input, output=x)

		return model
