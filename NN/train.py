from __future__ import print_function
from os import path
import time
import datetime
from mobilenet import get_model
from nn_utils import get_train_generator, get_valid_generator, get_callbacks
from image_transformations import rgb2hsv, rgb2hls


def get_start_date():
    return str(datetime.date.today()).replace('-', '_') + "_" + str(datetime.datetime.now().hour) + "_" + str(
        datetime.datetime.now().minute)


def train(classifier_type, train_images_path, valid_images_path, num_classes, input_shape, target_size,
          reserve_layers=15, epochs=500, preprocessing_function=None, log=None):
    start_time = time.time()
    model_path = path.join("../models/smaller_mobilenet",
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
    score_seg = model.evaluate_generator(
        generator= valid_generator,
        verbose=1,
        workers=8,
    )

    if log is not None:
        log.write(classifier_type + ':\nTraining time - %d seconds\nValidate Accuracy - %s\n\n\n' % (training_time, str(score_seg[1])))
        log.flush()

    model = None
    train_generator = None
    valid_generator = None
    callbacks = None