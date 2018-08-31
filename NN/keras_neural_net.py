from __future__ import print_function
from os import path
import datetime
import time

from mobilenet2 import get_model
from nn_utils import get_train_generator, get_valid_generator, get_callbacks
from image_transformations import rgb2hsv, rgb2hls

from image_transformations import rgb2yuv


def get_start_date():
    return str(datetime.date.today()).replace('-', '_') + "_" + str(datetime.datetime.now().hour) + "_" + str(
        datetime.datetime.now().minute)


def train(classifier_type, train_images_path, valid_images_path, num_classes, input_shape, target_size,
          reserve_layers=15, epochs=500, preprocessing_function=None):
    start_time = time.time()
    model_path = path.join("../models/mobilenet/mid_models",
                           get_start_date() + "_" + repr(epochs) + "_epochs_" + classifier_type + ".model")

    model = get_model(input_shape, num_classes=num_classes, reserve_layers=reserve_layers)

    train_generator = get_train_generator(train_images_path, batch_size=250, target_size=target_size, preprocessing_function=preprocessing_function)

    valid_generator = get_valid_generator(valid_images_path, target_size, preprocessing_function=preprocessing_function)

    print('Train indices: ' + str(train_generator.class_indices))
    print('Validate indices: ' + str(valid_generator.class_indices))

    callbacks = get_callbacks(validate_freq=10,
                              valid_generator=valid_generator,
                              model=model,
                              patience=5,
                              save_freq=10,
                              model_path=model_path)

    model.fit_generator(generator=train_generator,
                        epochs=epochs,
                        verbose=1,
                        workers=8,
                        callbacks=callbacks)

    training_time = time.time() - start_time
    print("Total training time is:", training_time)
    model.save(model_path)

    print("Evaluating...")
    scoreSeg = model.evaluate_generator(
        generator= valid_generator,
        verbose=1,
        workers=8,
    )

    return training_time, scoreSeg[1]


train(classifier_type='size_128_5_classes_hsv',
      train_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5',
      valid_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation',
      num_classes=5,
      input_shape=(128, 128, 3),
      target_size=(128, 128),
      reserve_layers=20,
      preprocessing_function=rgb2hsv)

train(classifier_type='size_128_5_classes_hsl',
      train_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5',
      valid_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation',
      num_classes=5,
      input_shape=(128, 128, 3),
      target_size=(128, 128),
      reserve_layers=20,
      preprocessing_function=rgb2hls)

