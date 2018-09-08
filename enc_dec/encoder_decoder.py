import json
import time

import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Lambda, Dropout, merge
import cv2
from keras.optimizers import Adam


def encoder_decoder(input_shape):
    return Sequential([
        Lambda(lambda x: x / 127.5 - 1.0, input_shape=input_shape),
        Convolution2D(8, 3, 3, activation='relu', border_mode='same'),
        Convolution2D(8, 3, 3, activation='relu', border_mode='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Dropout(0.2),
        Convolution2D(16, 3, 3, activation='relu', border_mode='same'),
        Convolution2D(16, 3, 3, activation='relu', border_mode='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Dropout(0.2),
        Convolution2D(32, 3, 3, activation='relu', border_mode='same'),
        Convolution2D(32, 3, 3, activation='relu', border_mode='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Dropout(0.2),
        Convolution2D(64, 3, 3, activation='relu', border_mode='same'),
        Convolution2D(64, 3, 3, activation='relu', border_mode='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Dropout(0.2),
        Convolution2D(128, 3, 3, activation='relu', border_mode='same'),
        Convolution2D(128, 3, 3, activation='relu', border_mode='same'),
        UpSampling2D(size=(2, 2)),
        Dropout(0.2),
        Convolution2D(64, 3, 3, activation='relu', border_mode='same'),
        Convolution2D(64, 3, 3, activation='relu', border_mode='same'),
        UpSampling2D(size=(2, 2)),
        Dropout(0.2),
        Convolution2D(32, 3, 3, activation='relu', border_mode='same'),
        Convolution2D(32, 3, 3, activation='relu', border_mode='same'),
        UpSampling2D(size=(2, 2)),
        Dropout(0.2),
        Convolution2D(16, 3, 3, activation='relu', border_mode='same'),
        Convolution2D(16, 3, 3, activation='relu', border_mode='same'),
        UpSampling2D(size=(2, 2)),
        Dropout(0.2),
        Convolution2D(8, 3, 3, activation='relu', border_mode='same'),
        Convolution2D(8, 3, 3, activation='relu', border_mode='same'),
        Convolution2D(8, (17, 17), strides=(20, 20)),
        Convolution2D(5, 1, 1, activation='relu')
    ])


if __name__ == "__main__":
    model = encoder_decoder((500, 500, 3))

    image = cv2.imread(
        "C:\Tomato_Classification_Project\Tomato_Classification_Project\cropped_data\size_500_stride_500/346.png", 1)
    label = json.loads(
        open("C:\Tomato_Classification_Project\Tomato_Classification_Project\encoder_decoder_train_set/346.txt",
             'r').read())
    model.compile(optimizer=Adam(lr=1.0e-5), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.fit(x=[image.reshape(1, 500, 500, 3)], y=[np.array(label).reshape(1, 24, 24, 5)], batch_size=1, epochs=100, verbose=1)
    t = time.time()
    y = model.predict(image.reshape(1, 500, 500, 3), verbose=1)
    print("time:", time.time() - t)
    # model.summary()
