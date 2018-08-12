import tflearn
from tflearn import ImageAugmentation
from tflearn.data_preprocessing import ImagePreprocessing

from tflearn.layers.conv import conv_2d, max_pool_2d, residual_block
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

    NN = conv_2d(NN, 64, 2, activation='relu')
    NN = max_pool_2d(NN, 2)
    NN = batch_normalization(NN)

    NN = tflearn.conv_2d(NN, 16, 3, regularizer='L2', weight_decay=0.0001)
    NN = tflearn.residual_block(NN, 5, 16)
    NN = tflearn.residual_block(NN, 1, 32, downsample=True)
    NN = tflearn.residual_block(NN, 5 - 1, 32)
    NN = tflearn.residual_block(NN, 1, 64, downsample=True)
    NN = tflearn.residual_block(NN, 5 - 1, 64)
    NN = tflearn.batch_normalization(NN)
    NN = tflearn.activation(NN, 'relu')
    NN = tflearn.global_avg_pool(NN)

    '''
    NN = conv_2d(NN, 64, 2, activation='relu')
    NN = max_pool_2d(NN, 2)
    NN = batch_normalization(NN)'''

    NN = fully_connected(NN, 1024, activation='relu', weights_init='xavier', bias_init='xavier')
    NN = dropout(NN, 0.5)

    NN = fully_connected(NN, 5, activation='softmax')

    NN = regression(NN, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')
    return tflearn.DNN(NN, tensorboard_dir="../logs/", tensorboard_verbose=0)
