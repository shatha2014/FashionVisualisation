
# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
#from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from VGG7 import VGG7
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from PIL import Image
import argparse

def train_model (path1, path2):
    # initialize the number of epochs to train for, base learning rate,
    # and batch size
    NUM_EPOCHS = 25
    INIT_LR = 1e-2
    BS = 32

    # grab the Fashion MNIST dataset
    print("[INFO] loading Instagram dataset")
    num_classes = 2
    # Define data path
    data_path_list = [path1, path2]
    #data_dir_list = os.listdir(data_path)

    img_data_list=[]
    labels_list = []

    # initialize the label names
    labelNames = ["Feminie+Elegant+Girly Style", "Trendy Style"]
    listcounter=0
    for dataset in data_path_list:
        img_list=os.listdir(data_path_list[listcounter])
        label = labelNames[listcounter]
        for item in img_list:
	    item_path = os.path.join(data_path_list[listcounter] + item)
            im = cv2.imread(item_path)
	    input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
	    input_img_resize=cv2.resize(input_img,(224,224))
	    img_data_list.append(im)
	    labels_list.append(label)
        listcounter+=1



    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255
    #print (img_data.shape)

    labels = np.array(labels_list)
    # print the count of number of samples for different classes
    print(np.unique(labels,return_counts=True))
    # convert class labels to on-hot encoding
    Y = np_utils.to_categorical(labels, num_classes)


    #Shuffle the dataset
    x,y = shuffle(img_data,Y, random_state=2)
    # Split the dataset
    trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2, random_state=2)

    # thano dim-ordering
    # 	num_samples x depth x rows x columns
    trainX = trainX.reshape((trainX.shape[0], 3, 224, 224))
    testX = testX.reshape((testX.shape[0], 3, 224, 224))

    # scale data to the range of [0, 1]
    #trainX = trainX.astype("float32") / 255.0
    #testX = testX.astype("float32") / 255.0

    # one-hot encode the training and testing labels
   # trainY = np_utils.to_categorical(trainY, 2)
    #testY = np_utils.to_categorical(testY, 2)



    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
    model = VGG7.build(width=224, height=224, depth=3, classes=2)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    # train the network
    print("[INFO] training model...")
    H = model.fit(trainX, trainY,
                  validation_data=(testX, testY),
                  batch_size=BS, nb_epoch=NUM_EPOCHS)

    # make predictions on the test set
    preds = model.predict(testX)
    model.save('Instagram_2styles.h5')

    # show a nicely formatted classification report
    print("[INFO] evaluating network...")
    print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1),
                                target_names=labelNames))

    # plot the training loss and accuracy
    N = NUM_EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path1', type=str, metavar='Path1',
                        help='path1 of folder with images')
    parser.add_argument('--folder_path2', type=str, metavar='Path2',
                        help='path2 of folder with images')
    Path1 = parser.parse_args().folder_path1
    Path2 = parser.parse_args().folder_path2
    print(Path1, Path2)
    train_model(Path1, Path2)
