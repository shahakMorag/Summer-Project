import json
import time

import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Lambda, Dropout, SeparableConv2D, DepthwiseConv2D, BatchNormalization
import cv2
from keras.optimizers import Adam


def encoder_decoder(input_shape):
    depthwise_depth = 8
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0, input_shape=input_shape))
    model.add(Convolution2D(32, (5, 5), padding="same", activation="relu"))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D((2, 2)))
    for i in range(depthwise_depth):
        model.add(DepthwiseConv2D(kernel_size=(5, 5), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Convolution2D(kernel_size=(1,1), filters=64, activation='relu'))
        if i % 3 == 2:
            model.add(MaxPooling2D((2, 2)))

    model.add(Convolution2D(kernel_size=(15, 15), strides=(2, 2), filters=5, activation='relu'))
    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()
    return model


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
