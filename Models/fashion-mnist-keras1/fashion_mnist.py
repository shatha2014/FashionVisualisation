# USAGE
# python fashion_mnist.py

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from pyimagesearch.fullVGG import fullVGGNet
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils.mnist_reader import load_mnist
from keras import callbacks

# initialize the number of epochs to train for, base learning rate,
# and batch size
NUM_EPOCHS = 25
INIT_LR = 1e-2
BS = 32

# grab the Fashion MNIST dataset
print("[INFO] loading Fashion MNIST...")
trainX, trainY = load_mnist('data/fashion', kind='train')
testX, testY = load_mnist('data/fashion', kind='t10k')


# thano dim-ordering
# 	num_samples x depth x rows x columns
trainX = trainX.reshape((trainX.shape[0], 1, 28, 28))
testX = testX.reshape((testX.shape[0], 1, 28, 28))

# scale data to the range of [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# one-hot encode the training and testing labels
trainY = np_utils.to_categorical(trainY, 10)
testY = np_utils.to_categorical(testY, 10)

# initialize the label names
labelNames = ["top", "trouser", "pullover", "dress", "coat",
	"sandal", "shirt", "sneaker", "bag", "ankle boot"]

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
model = fullVGGNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# save model after each epoch
filepath = "saved_epoch_model/fullVGG-model-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
callback_list = [checkpoint]

# train the network
print("[INFO] training model...")
H = model.fit(trainX, trainY,
	validation_data=(testX, testY),
	batch_size=BS, nb_epoch=NUM_EPOCHS, callbacks=callback_list)

# make predictions on the test set
preds = model.predict(testX)
model.save('fmnist_fullVGG.h5')

# show a nicely formatted classification report
print("[INFO] evaluating network...")
#print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1),
#	target_names=labelNames))

# plot the training loss and accuracy
#N = NUM_EPOCHS
#plt.style.use("ggplot")
#plt.figure()
#plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
#plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
#plt.title("Training Loss and Accuracy on Dataset")
#plt.xlabel("Epoch #")
#plt.ylabel("Loss/Accuracy")
#plt.legend(loc="lower left")
#plt.savefig("plot.png")

# initialize our list of output images
#images = []

# randomly select a few testing fashion items
#for i in np.random.choice(np.arange(0, len(testY)), size=(16,)):
	# classify the clothing
	#probs = model.predict(testX[np.newaxis, i])
	#prediction = probs.argmax(axis=1)
	#label = labelNames[prediction[0]]
 

	#image = (testX[i][0] * 255).astype("uint8")


	# initialize the text label color as green (correct)
	#color = (0, 255, 0)

	# otherwise, the class label prediction is incorrect
	#if prediction[0] != np.argmax(testY[i]):
	#	color = (0, 0, 255)
 
	# merge the channels into one image and resize the image from
	# 28x28 to 96x96 so we can better see it and then draw the
	# predicted label on the image
	#image = cv2.merge([image] * 3)
	#image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
	#cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
	#	color, 2)

	# add the image to our list of output images
	#images.append(image)

# construct the montage for the images
#montage = build_montages(images, (96, 96), (4, 4))[0]

# show the output montage
#cv2.imshow("Fashion MNIST", montage)
#cv2.waitKey(0)
