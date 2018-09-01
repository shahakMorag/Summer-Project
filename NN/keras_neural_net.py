from __future__ import print_function
from os import path
import time
import datetime
from mobilenet2 import get_model
from nn_utils import get_train_generator, get_valid_generator, get_callbacks
from image_transformations import rgb2hsv, rgb2hls


def get_start_date():
    return str(datetime.date.today()).replace('-', '_') + "_" + str(datetime.datetime.now().hour) + "_" + str(
        datetime.datetime.now().minute)


def train(classifier_type, train_images_path, valid_images_path, num_classes, input_shape, target_size,
          reserve_layers=15, epochs=500, preprocessing_function=None, log=None):
    start_time = time.time()
    model_path = path.join("../models/mobilenet/all_models",
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

    model = None
    train_generator = None
    valid_generator = None
    callbacks = None
    log.flush()


output = open('../logs/details/' + get_start_date() + '.txt', 'w')

# RGB models ------------------------------------------------------------------------------------------------------------------------

preprocessing_function = None
'''
# size 100x100
train(classifier_type='rgb_size_100',
      train_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_100_skip_32_categories_5',
      valid_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation_100',
      num_classes=5,
      input_shape=(100, 100, 3),
      target_size=(100, 100),
      reserve_layers=20,
      preprocessing_function=preprocessing_function,
      log=output)

# size 128x128
train(classifier_type='rgb_size_128',
      train_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5',
      valid_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation_128',
      num_classes=5,
      input_shape=(128, 128, 3),
      target_size=(128, 128),
      reserve_layers=20,
      preprocessing_function=preprocessing_function,
      log=output)

# size 180x180
train(classifier_type='rgb_size_180',
      train_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_180_skip_16_categories_5',
      valid_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation_180',
      num_classes=5,
      input_shape=(180, 180, 3),
      target_size=(180, 180),
      reserve_layers=20,
      preprocessing_function=preprocessing_function,
      log=output)
'''
## 2-round 128x128
# size 128x128 round 1
train(classifier_type='rgb_128_round_1',
      train_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\stem_tomato',
      valid_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\stem_tomato_validation',
      num_classes=3,
      input_shape=(128, 128, 3),
      target_size=(128, 128),
      reserve_layers=10,
      preprocessing_function=preprocessing_function,
      log=output)

# size 128x128 round 2
train(classifier_type='rgb_128_round_2',
      train_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\leaf_net_training',
      valid_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\leaf_validation',
      num_classes=3,
      input_shape=(128, 128, 3),
      target_size=(128, 128),
      reserve_layers=10,
      preprocessing_function=preprocessing_function,
      log=output)

# HSV models ------------------------------------------------------------------------------------------------------------------------

preprocessing_function = rgb2hsv

# size 100x100
train(classifier_type='hsv_size_100',
      train_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_100_skip_32_categories_5',
      valid_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation_100',
      num_classes=5,
      input_shape=(100, 100, 3),
      target_size=(100, 100),
      reserve_layers=20,
      preprocessing_function=preprocessing_function,
      log=output)

# size 128x128
train(classifier_type='hsv_size_128',
      train_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5',
      valid_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation_128',
      num_classes=5,
      input_shape=(128, 128, 3),
      target_size=(128, 128),
      reserve_layers=20,
      preprocessing_function=preprocessing_function,
      log=output)

# size 180x180
train(classifier_type='hsv_size_180',
      train_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_180_skip_16_categories_5',
      valid_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation_180',
      num_classes=5,
      input_shape=(180, 180, 3),
      target_size=(180, 180),
      reserve_layers=20,
      preprocessing_function=preprocessing_function,
      log=output)

## 2-round 128x128
# size 128x128 round 1
train(classifier_type='hsv_128_round_1',
      train_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\stem_tomato',
      valid_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\stem_tomato_validation',
      num_classes=3,
      input_shape=(128, 128, 3),
      target_size=(128, 128),
      reserve_layers=10,
      preprocessing_function=preprocessing_function,
      log=output)

# size 128x128 round 2
train(classifier_type='hsv_128_round_2',
      train_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\leaf_net_training',
      valid_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\leaf_validation',
      num_classes=3,
      input_shape=(128, 128, 3),
      target_size=(128, 128),
      reserve_layers=10,
      preprocessing_function=preprocessing_function,
      log=output)

# HLS models ------------------------------------------------------------------------------------------------------------------------

preprocessing_function = rgb2hls

# size 100x100
train(classifier_type='hls_size_100',
      train_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_100_skip_32_categories_5',
      valid_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation_100',
      num_classes=5,
      input_shape=(100, 100, 3),
      target_size=(100, 100),
      reserve_layers=20,
      preprocessing_function=preprocessing_function,
      log=output)

# size 128x128
train(classifier_type='hls_size_128',
      train_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5',
      valid_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation_128',
      num_classes=5,
      input_shape=(128, 128, 3),
      target_size=(128, 128),
      reserve_layers=20,
      preprocessing_function=preprocessing_function,
      log=output)

# size 180x180
train(classifier_type='hls_size_180',
      train_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_180_skip_16_categories_5',
      valid_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation_180',
      num_classes=5,
      input_shape=(180, 180, 3),
      target_size=(180, 180),
      reserve_layers=20,
      preprocessing_function=preprocessing_function,
      log=output)

## 2-round 128x128
# size 128x128 round 1
train(classifier_type='hls_128_round_1',
      train_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\stem_tomato',
      valid_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\stem_tomato_validation',
      num_classes=3,
      input_shape=(128, 128, 3),
      target_size=(128, 128),
      reserve_layers=10,
      preprocessing_function=preprocessing_function,
      log=output)

# size 128x128 round 2
train(classifier_type='hls_128_round_2',
      train_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\leaf_net_training',
      valid_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\leaf_validation',
      num_classes=3,
      input_shape=(128, 128, 3),
      target_size=(128, 128),
      reserve_layers=10,
      preprocessing_function=preprocessing_function,
      log=output)

output.close()
