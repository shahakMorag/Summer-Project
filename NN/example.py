import tflearn
from tflearn.data_preprocessing import ImagePreprocessing

from tflearn.layers.conv import conv_2d, max_pool_2d, conv_3d, max_pool_3d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization

import tflearn.datasets.mnist as mnist

X, Y, test_x, test_y = mnist.load_data(one_hot=True)

X = X.reshape([-1, 28, 28, 1])
test_x = test_x.reshape([-1, 28, 28, 1])


def custom_network():
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    NN = input_data(shape=[None, 28, 28, 1],
                    data_preprocessing=img_prep,
                    name='input')

    NN = conv_2d(NN, 32, 4, activation='relu')
    NN = max_pool_2d(NN, 2)
    NN = batch_normalization(NN)

    NN = conv_2d(NN, 64, 2, activation='relu')
    NN = max_pool_2d(NN, 2)
    NN = batch_normalization(NN)

    NN = conv_2d(NN, 64, 2, activation='relu')
    NN = max_pool_2d(NN, 2)
    NN = batch_normalization(NN)

    NN = conv_2d(NN, 64, 2, activation='relu')
    NN = max_pool_2d(NN, 2)
    NN = batch_normalization(NN)

    NN = fully_connected(NN, 1024, activation='relu', weights_init='xavier', bias_init='xavier')
    NN = dropout(NN, 0.5)

    NN = fully_connected(NN, 10, activation='softmax')

    NN = regression(NN, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')
    return tflearn.DNN(NN)


model = custom_network()

model.fit({'input': X}, {'targets': Y},
          n_epoch=3,
          validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=500,
          show_metric=True, run_id='mnist')

model.save('tflearncnn.model')