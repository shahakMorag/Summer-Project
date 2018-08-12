import tflearn
from tflearn.data_preprocessing import ImagePreprocessing

from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization


def custom_network():
    '''img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()'''

    NN = input_data(shape=[None, 128, 128, 3],
                    # data_preprocessing=img_prep,
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

    NN = fully_connected(NN, 128, activation='relu', weights_init='xavier', bias_init='xavier')
    NN = dropout(NN, 0.5)

    NN = fully_connected(NN, 5, activation='softmax')

    NN = regression(NN, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')
    return tflearn.DNN(NN)

