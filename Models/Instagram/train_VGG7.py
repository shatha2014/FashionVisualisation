
import numpy as np
import os
from keras.preprocessing import image
from VGG7 import VGG7_model
from keras.optimizers import SGD
from keras import backend as K
K.set_image_dim_ordering('th')

img_width, img_height = 224, 224

# initialize the number of epochs to train for, base learning rate,
# and batch size
NUM_EPOCHS = 25
INIT_LR = 1e-2
BS = 32
num_images = 3815
num_val_images = 400

model = VGG7_model.build(width=img_width, height=img_height, depth=3, classes=2)
opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


train_datagen = image.ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory('Instagram_dataset/train',
                                              target_size = (img_width, img_height),
                                              batch_size = BS)
val_gen = train_datagen.flow_from_directory('Instagram_dataset/Validation',
                                            target_size = (img_width, img_height),
                                            batch_size = BS)


model.fit_generator(train_gen,
                    samples_per_epoch=num_images,
                    validation_data = val_gen,
                    nb_val_samples = num_val_images,
                    nb_epoch=NUM_EPOCHS)

model.save('VGG7.h5')
