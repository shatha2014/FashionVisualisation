
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
#sys.path.insert(0, '/home/isak/Programming/github projects/FashionVisualisation/Models/fashion-mnist-keras1/')

from utils.mnist_reader import load_mnist

### Load model




def analysis(model_path, layer_name, class_ind):
    model = load_model(model_path)
    num_filters = model.get_layer(name=layer_name).get_config()['nb_filter']
    csv_path = '../DeconvNets/results_minivgg_FashionMNIST/class_{}/feat_inf.cvs'.format(class_ind)

    testX0 = load_img(class_ind)

    for i, img_array in enumerate(testX0):
        feature_influence = [i]
        img_array = img_array[np.newaxis, :]
        class_prob = class_probability(model, img_array, class_ind)

        for j in range(num_filters):
            feature = visualize(model=model, data=img_array, layer_name=layer_name, feature_to_visualize=j, visualize_mode='all')
            img_without_feature = img_array - feature
            class_prob_wo_feat = class_probability(model, img_without_feature, class_ind)
            difference = class_prob - class_prob_wo_feat
            feature_influence.append(difference.item())

        if i == 0:
            csv_file = open(csv_path, 'wt')
            filewriter = csv.writer(csv_file, delimiter=',')
            filewriter.writerow(feature_influence)
        else:
            filewriter.writerow(feature_influence)


def class_probability(model, image, class_ind):
    f = K.function([model.layers[0].input], [model.layers[-1].output])
    return f([image])[0][0][class_ind]

def load_img(class_ind):
    testX, testY = load_mnist(os.path.dirname(os.path.abspath(__file__)) + '/data/fashion', kind='t10k')
    testX = testX.reshape((testX.shape[0], 1, 28, 28))
    testX = testX.astype("float32") / 255.0
    test_mask = np.isin(testY, [class_ind])
    testX0 = testX[test_mask]
    #testX0 = testX0[0]
    #testX0 = testX0[np.newaxis, :]

    return testX0

def save_deconv_img(image, image_ind, class_ind, filter, model, layer_name):
    deconv = visualize(model=model, data=image, layer_name=layer_name, feature_to_visualize=i,
                        visualize_mode='all')
    deconv = deconv - deconv.min()
    deconv *= 1.0 / (deconv.max() + 1e-8)
    deconv = np.multiply(deconv, 255)
    uint8_deconv = (deconv).astype(np.uint8)
    feature = Image.fromarray(uint8_deconv, 'L')
    #img = Image.fromarray(image, 'RGB')
    try:
        os.makedirs('../DeconvNets/results_minivgg_FashionMNIST/class_{}/img_{}/'.format(class_ind, image_ind))
    except FileExistsError:
        pass
    feature.save('../DeconvNets/results_minivgg_FashionMNIST/class_{}/img_{}/{}_{}.png'.format(class_ind, image_ind, layer_name, filter))
    #img.save('../DeconvNets/results_minivgg_FashionMNIST/class_{}/img_{}/img.png'.format(class_ind, image_ind, layer_name, filter))

#model = load_model('/home/isak/Programming/github projects/FashionVisualisation/Models/fashion-mnist-keras1/fashion_mnist_model_miniVGG_without_Normalisation.h5')
#img=load_img(0)
#for i in range(20):
#    save_deconv_img(img, 0, 0, i, model, 'convolution2d_4')


analysis('/home/isak/Programming/github projects/FashionVisualisation/Models/fashion-mnist-keras1/fashion_mnist_model_miniVGG_without_Normalisation.h5',
        layer_name=layer_name, class_ind=class_ind)

