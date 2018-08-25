from __future__ import print_function
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from mobilenet2 import get_model
import datetime
from keras_generators import get_train_generator, get_valid_generator

# input image dimensions and parameters


dir_to_save_model = "../models/mobilenet"
start_date = str(datetime.date.today()).replace('-', '_') + "_" + str(datetime.datetime.now().hour) + "_" + str(datetime.datetime.now().minute)

input_shape = (128, 128, 3)
patience = 10
batch_size = 100
epochs = 100
seed = 10

train_images_path = 'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5'
valid_images_path = 'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation'

#
model = get_model(input_shape, num_classes=5)

print('Starting to fit the model...')

# ------------------------------------------ training data set ------------------------------------------

train_generator = get_train_generator(train_images_path, batch_size)
print(train_generator.class_indices)

# ------------------------------------------ validation data set ----------------------------------------

valid_generator = get_valid_generator(valid_images_path)

# ------------------------------------------ callbacks --------------------------------------------------

# TODO: add evaluate callbacks and monitor callbacks
early_stop = EarlyStopping('acc', patience=patience)
reduce_lr = ReduceLROnPlateau('acc', factor=0.9,
                              patience=2, verbose=1)

callbacks = [early_stop, reduce_lr]

model.fit_generator(
    generator=train_generator,
    #validation_data=valid_generator,
    epochs=epochs,
    verbose=1,
    workers=8,
    callbacks=callbacks)

model.save(dir_to_save_model + "/" + start_date + "_" + repr(epochs) + "_epochs.model")

print("Evaluating...")
scoreSeg = model.evaluate_generator(
    generator=valid_generator,
    verbose=1,
    workers=8,
)

print('Loss: ' + repr(scoreSeg[0]) + ', Acc: ' + repr(scoreSeg[1]))

