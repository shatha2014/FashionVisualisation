
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



def analysis(model_path, layer_name, class_ind, list_images):
    model = load_model(model_path)
    num_filters = model.get_layer(name=layer_name).get_config()['nb_filter']
    csv_path = '../../DeconvNets/results_only_features/class_{}/feat_inf.csv'.format(class_ind)
    testX0 = load_img(list_images, class_ind)

    for i, img_array in enumerate(testX0):
        feature_influence = [list_images[i]]
        img_array = img_array[np.newaxis, :]
        class_prob = model.predict(img_array)[0][class_ind]

        for j in range(num_filters):
            deconv = visualize(model=model, data=img_array, layer_name=layer_name, feature_to_visualize=j, visualize_mode='all')

            deconv = deconv - deconv.min()
            deconv *= 1.0 / (deconv.max() + 1e-8)

            img_without_feature = img_array - deconv

            img_without_feature = img_without_feature - img_without_feature.min()
            img_without_feature *= 1.0 / (img_without_feature.max() + 1e-8)
            feature = img_array-img_without_feature


            class_prob_wo_feat = model.predict(feature)[0][class_ind]

            difference = class_prob - class_prob_wo_feat


            feature_influence.append(difference.item())

        if i == 0:
            try:
                os.mkdir('../../DeconvNets/results_only_features/class_{}'.format(class_ind))
            except:
                pass
            csv_file = open(csv_path, 'at')

            filewriter = csv.writer(csv_file, delimiter=',')
            header = [x-1 for x in range(65)]

            filewriter.writerow(header)
            filewriter.writerow(feature_influence)
        else:
            filewriter.writerow(feature_influence)
        print('image {} done'.format(i))

def load_img(images, class_ind):
    testX, testY = load_mnist(os.path.dirname(os.path.abspath(__file__)) + '/data/fashion', kind='t10k')
    testX = testX.reshape((testX.shape[0], 1, 28, 28))
    testX = testX.astype("float32") / 255.0
    test_mask = np.isin(testY, [class_ind])
    testX = testX[test_mask]

    img = []
    for image in images:
        img.append(testX[image])
    return img

def save_deconv_img(image, image_ind, class_ind, filter, model, layer_name):
    img_array = image[np.newaxis, :]
    deconv = visualize(model=model, data=img_array, layer_name=layer_name, feature_to_visualize=filter,
                        visualize_mode='all')
    deconv = deconv - deconv.min()
    deconv *= 1.0 / (deconv.max() + 1e-8)
    deconv = np.multiply(deconv, 255)
    uint8_deconv = (deconv).astype(np.uint8)
    feature = Image.fromarray(uint8_deconv, 'L')

    img_array = np.multiply(img_array, 255)
    uint8_img_array = img_array.astype(np.uint8)
    img = Image.fromarray(uint8_img_array[0][0], 'L')

    try:
        os.makedirs('../../DeconvNets/results_minivgg_FashionMNIST/class_{}/img_{}/'.format(class_ind, image_ind))
    except:
        pass

    feature.save('../../DeconvNets/results_minivgg_FashionMNIST/class_{}/img_{}/{}_{}.png'.format(class_ind, image_ind, layer_name, filter))

    exist = os.path.isfile('../../DeconvNets/results_minivgg_FashionMNIST/class_{}/img_{}/img.png'.format(class_ind, image_ind, layer_name, filter))
    if not exist:
        img.save('../../DeconvNets/results_minivgg_FashionMNIST/class_{}/img_{}/img.png'.format(class_ind, image_ind, layer_name, filter))


def save_images(img_index, filter_index, class_index):
    model = load_model(os.path.dirname(os.path.abspath(__file__)) + '/FM_miniVGG.h5')
    img = load_img(img_index, class_index)
    for i, image in enumerate(img):
        for filter in filter_index:
            save_deconv_img(image, img_index[i], class_index, filter, model, 'convolution2d_4')

for C in [2,4,6]:
    print(C)
    if C == 2:
        save_images(img_index = [ 5, 6, 7, 8, 9], filter_index=[26, 8, 7, 60, 39, 37, 10, 62, 1, 16], class_index=C)
    elif C == 4:
        save_images(img_index = [ 5, 6, 7, 8, 9], filter_index=[47, 35, 57, 4, 11, 21, 32, 27, 54, 50], class_index=C)
    else:
        save_images(img_index=[ 5, 6, 7, 8, 9], filter_index=[34, 54, 58, 15, 33, 43, 17, 51, 37, 3], class_index=C)
  #  elif C == 5:
        #save_images(img_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], filter_index=[45, 53, 22, 52, 11, 20, 62, 27, 13, 37], class_index=C)
   # elif C == 7:
        #save_images(img_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], filter_index=[39, 11, 45, 31, 21, 37, 17, 49, 25, 63], class_index=C)
   # elif C == 8:
       # save_images(img_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], filter_index=[42, 60, 8, 36, 9, 2, 44, 10, 22, 32], class_index=C)
    #else:
   #     save_images(img_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], filter_index=[11, 8, 40, 45, 53, 37, 63, 62, 24, 25], class_index=C)



#lis = [x for x in range(100)]
#for C in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
   # analysis(os.path.dirname(os.path.abspath(__file__)) + '/FM_miniVGG.h5', layer_name='convolution2d_4', class_ind=C, list_images=lis)

