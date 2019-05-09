
import numpy as np
import sys
from PIL import Image
from keras.activations import *
from keras.models import Model, Sequential
import keras.backend as K
from keras.models import load_model
import os
from keras.utils import np_utils
from keras import backend as K
from imutils import build_montages
import cv2
from utils.mnist_reader import load_mnist
import argparse

def Extract_class(class_number):
    testX, testY = load_mnist('data/fashion', kind='t10k')

    test_mask = np.isin(testY, [class_number])
    testX_sample, testY_sample = testX[test_mask], np.array(testY[test_mask]==class_number)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_number', type=int, metavar='CN',
                        help='number of which class you want to extract ')
    class_number = parser.parse_args().class_number
    print(class_number)
    Extract_class(class_number)