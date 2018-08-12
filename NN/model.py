import tflearn
from tflearn import ImageAugmentation
from tflearn.data_preprocessing import ImagePreprocessing

from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization


def custom_network():
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)

    # img_prep = ImagePreprocessing()

    NN = input_data(shape=[None, 128, 128, 3],
                    # data_preprocessing=img_prep,
                    data_augmentation=img_aug,
                    name='input')

    NN = conv_2d(NN, 32, 2, activation='relu')
    NN = max_pool_2d(NN, 2)
    NN = batch_normalization(NN)

    '''NN = conv_2d(NN, 64, 2, activation='relu')
    NN = max_pool_2d(NN, 2)
    NN = batch_normalization(NN)

    NN = conv_2d(NN, 64, 2, activation='relu')
    NN = max_pool_2d(NN, 2)
    NN = batch_normalization(NN)'''

    NN = fully_connected(NN, 1024, activation='relu', weights_init='xavier', bias_init='xavier')
    NN = dropout(NN, 0.5)

    NN = fully_connected(NN, 5, activation='softmax')

    NN = regression(NN, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')
    return tflearn.DNN(NN)

