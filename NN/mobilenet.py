import numpy as np
import keras
from keras import backend as K
from keras.layers.core import Dense
from keras.engine import InputLayer
from keras.optimizers import Adam
from keras.applications import imagenet_utils
from keras.models import Model


def get_model(input_shape, num_classes):
    mobile = keras.applications.mobilenet.MobileNet(input_shape=input_shape)

    x = mobile.layers[-6].output
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=mobile.input, outputs=predictions)

    for layer in model.layers[:-5]:
        layer.trainable = False

    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


get_model((128,128,3),5)
