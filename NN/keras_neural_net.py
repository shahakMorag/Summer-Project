from __future__ import print_function
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.callbacks import Callback
from mobilenet2 import get_model
import datetime
import time
from nn_utils import get_train_generator, get_valid_generator
'''
import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
'''

class ValidateCallback(Callback):

    def __init__(self, p, valid_generator, model):
        self.param = p
        self.valid_generator=valid_generator
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        if not epoch % self.param == 0:
            return

        print("Evaluating...")
        scoreSeg = self.model.evaluate_generator(
            generator=self.valid_generator,
            verbose=1,
            workers=8,
        )

        print('Loss: ' + repr(scoreSeg[0]) + ', Acc: ' + repr(scoreSeg[1]))


def train(classifier_type, train_images_path, valid_images_path, num_classes):
    # input image dimensions and parameters
    start_time = time.time()
    dir_to_save_model = "../models/mobilenet/2round"
    start_date = str(datetime.date.today()).replace('-', '_') + "_" + str(datetime.datetime.now().hour) + "_" + str(datetime.datetime.now().minute)

    input_shape = (128, 128, 3)
    patience = 20
    batch_size = 250
    epochs = 500
    validate_freq = 5

    model = get_model(input_shape, num_classes=num_classes)

    print('Starting to fit the model...')

    # ------------------------------------------ training data set ------------------------------------------

    train_generator = get_train_generator(train_images_path, batch_size)

    # ------------------------------------------ validation data set ----------------------------------------

    valid_generator = get_valid_generator(valid_images_path)

    print('Train indices: ' + str(train_generator.class_indices))
    print('Validate indices: ' + str(valid_generator.class_indices))

    # ------------------------------------------ callbacks --------------------------------------------------

    valid_callback = ValidateCallback(validate_freq, valid_generator, model)
    early_stop = EarlyStopping('acc', patience=patience)
    reduce_lr = ReduceLROnPlateau('acc', factor=0.6,
                                  patience=2, verbose=1)

    tensorboard = TensorBoard(log_dir='../logs',
                              histogram_freq=0,
                              batch_size=32,
                              write_graph=True,
                              write_grads=False,
                              write_images=False,
                              embeddings_freq=0,
                              embeddings_layer_names=None,
                              embeddings_metadata=None,
                              embeddings_data=None)

    callbacks = [valid_callback, early_stop, reduce_lr, tensorboard]

    model.fit_generator(
        generator=train_generator,
        epochs=epochs,
        verbose=1,
        workers=8,
        callbacks=callbacks)

    print("Total training time is:", time.time() - start_time)
    model.save(dir_to_save_model + "/" + start_date + "_" + repr(epochs) + "_epochs_" + classifier_type + ".model")


train(classifier_type='round_1_3_classes',
      train_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\stem_tomato',
      valid_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\stem_tomato_validation',
      num_classes=3)

train(classifier_type='round_2_3_classes',
      train_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\leaf_net_training',
      valid_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\leaf_validation',
      num_classes=3)

train(classifier_type='round_1_5_classes',
      train_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5',
      valid_images_path='C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation',
      num_classes=5)


