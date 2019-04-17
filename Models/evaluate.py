from keras.models import load_model, Sequential
from keras.preprocessing import image



img_width, img_height = 224, 224


model = load_model('VGG7_2Styles.h5')


train_datagen = image.ImageDataGenerator(rescale=1./255)
val_gen = train_datagen.flow_from_directory('Instagram_dataset/Validation',
                                            target_size = (img_width, img_height),
                                            batch_size = 32)

model.evaluate_generator(val_gen)