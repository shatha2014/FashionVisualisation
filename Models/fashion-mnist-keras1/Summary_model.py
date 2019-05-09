
import numpy as np
import sys
import time
from PIL import Image
from keras.layers import (
        Input,
        InputLayer,
        Flatten,
        Activation,
        Dense)
from keras.layers.convolutional import (
        Convolution2D,
        MaxPooling2D)
from keras.activations import *
from keras.models import Model, Sequential
from keras.applications import vgg16, imagenet_utils
import keras.backend as K
from keras.models import load_model
import os
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from imutils import build_montages
import numpy as np
import cv2
from utils.mnist_reader import load_mnist

model = load_model('FM_miniVGG.h5')
print(model.summary())
#trainX, trainY = load_mnist('data/fashion', kind='train')
#testX, testY = load_mnist('data/fashion', kind='t10k')
#trainX = trainX.reshape((trainX.shape[0], 1, 28, 28))
#testX = testX.reshape((testX.shape[0], 1, 28, 28))
#testX = testX.astype("float32") / 255.0
#trainX = trainX.astype("float32") / 255.0
#trainY = np_utils.to_categorical(trainY, 10)
#scores = model.evaluate(trainX, trainY, batch_size =32, verbose=1)
#print(model.metrics_names)
#print(scores)

#test_mask = np.isin(testY, [0])
#testX0, testY0 = testX[test_mask], np.array(testY[test_mask]==0)
#print(len(testX0))
#print(testY0)