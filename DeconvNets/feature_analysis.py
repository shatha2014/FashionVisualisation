
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

### Load model



### dif_list = []
###for each image in validation
    ### for each filter i
        ### feature = Deconv(image, filter= i)
        ### pred = cnn(image)
        ### pred_after = cnn(feature)
        ### dif = pred - pred_after
        ### dif_list[i] += dif

# normalize dif_list

def analysis(model_path, images_path, layer_name, class_ind):
    model = load_model(model_path)
    num_filters = model.get_layer(name=layer_name).get_config()['nb_filter']
    valid_images = [".jpg", ".gif", ".png", ".tga"]
    feature_influence = [0 for _ in range(num_filters)]
    num_images = 0

    for image in os.listdir(images_path):
        #To skip non image files
        ext = os.path.splitext(image)[1]
        if ext.lower() not in valid_images:
            continue

        img = Image.open(os.path.join(images_path, image))
        img_array = np.array(img)
        img_array = img_array[:, :, np.newaxis]
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = img_array[np.newaxis, :]
        img_array = img_array.astype(np.float)
        img_array = np.divide(img_array, 255)
        class_prob = class_probability(model, img_array, class_ind)
        for i in range(num_filters):
            feature = visualize(model=model, data=img_array, layer_name=layer_name, feature_to_visualize=i, visualize_mode='all')
            img_without_feature = img_array - feature
            class_prob_wo_feat  = class_probability(model, img_without_feature, class_ind)
            difference = class_prob - class_prob_wo_feat
            feature_influence[i] += difference
            print(image, ' filter=', i)
        num_images += 1

    feature_influence[:] = [x / num_images for x in feature_influence]

    return feature_influence

def class_probability(model, image, class_ind):
    f = K.function([model.layers[0].input], [model.layers[-1].output])
    return f([image])[0][0][class_ind]

feat_infl = analysis('/home/isak/Programming/github projects/FashionVisualisation/Models/fashion-mnist-keras1/fashion_mnist_model_miniVGG_without_Normalisation.h5',
         images_path='/home/isak/Programming/github projects/FashionVisualisation/DeconvNets/fashion_mnist_examples', layer_name='convolution2d_4', class_ind=0)

print(feat_infl)