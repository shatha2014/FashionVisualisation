
import argparse
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
from Deconvnetkeras import visualize
import csv
#import sys

from .activation_maximization import visualize_activation_with_losses
from .activation_maximization import visualize_activation

from .saliency import visualize_saliency_with_losses
from .saliency import visualize_saliency
#sys.path.insert(0, '/home/isak/Programming/github projects/FashionVisualisation/Models/fashion-mnist-keras1/')

from utils.mnist_reader import load_mnist
model= load_model('FM_miniVGG.h5')
visualize_saliency(model, layer_idx, filter_indices, seed_input, backprop_modifier=None, \
    grad_modifier="absolute")