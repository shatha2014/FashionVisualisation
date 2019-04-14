# USAGE
# python fashion_mnist.py

# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
#from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from PIL import Image

# initialize the number of epochs to train for, base learning rate,
# and batch size
NUM_EPOCHS = 25
INIT_LR = 1e-2
BS = 32

# grab the Fashion MNIST dataset
print("[INFO] loading Instagram dataset")
num_classes = 2
PATH = os.getcwd("/Instagram_dataset/")
# Define data path
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)

img_data_list=[]
labels_list = []

# initialize the label names
labelNames = ["Trendy Style", "Feminie+Elegant+Girly Style"]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+"/"+ dataset)
	print ('Loading the images of dataset-'+'{}\n'.format(dataset))
	label = labels_name[dataset]
	for img in img_list:
		input_img=cv2.imread(data_path + "/"+ dataset + "/"+ img )
		img_data_list.append(input_img)
		labels_list.append(label)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)

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
trainX = trainX.reshape((trainX.shape[0], 1, 224, 224))
testX = testX.reshape((testX.shape[0], 1, 224, 224))

# scale data to the range of [0, 1]
#trainX = trainX.astype("float32") / 255.0
#testX = testX.astype("float32") / 255.0

# one-hot encode the training and testing labels
#trainY = np_utils.to_categorical(trainY, 2)
#testY = np_utils.to_categorical(testY, 2)



# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
model = fullVGGNet.build(width=224, height=224, depth=3, classes=2)
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

# initialize our list of output images
images = []

# randomly select a few testing fashion items
for i in np.random.choice(np.arange(0, len(testY)), size=(16,)):
    # classify the clothing
    probs = model.predict(testX[np.newaxis, i])
    prediction = probs.argmax(axis=1)
    label = labelNames[prediction[0]]

    image = (testX[i][0] * 255).astype("uint8")

    # initialize the text label color as green (correct)
    color = (0, 255, 0)

    # otherwise, the class label prediction is incorrect
    if prediction[0] != np.argmax(testY[i]):
        color = (0, 0, 255)

    # merge the channels into one image and resize the image from
    # 28x28 to 96x96 so we can better see it and then draw the
    # predicted label on the image
    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                color, 2)

    # add the image to our list of output images
    images.append(image)

# construct the montage for the images
montage = build_montages(images, (96, 96), (4, 4))[0]

# show the output montage
cv2.imshow("Instagram", montage)
cv2.waitKey(0)