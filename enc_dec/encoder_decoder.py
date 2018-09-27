import json
import time

import numpy as np
import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Convolution2D, MaxPooling2D, Lambda, Dropout, BatchNormalization, Conv2DTranspose, \
    Dense, Flatten, Reshape, UpSampling2D, AvgPool2D, GaussianNoise
import cv2
from keras.optimizers import Adadelta
from keras import backend as K

def encoder_decoder(input_shape):
    return Sequential([
        Lambda(lambda x: x / 255.0, input_shape=input_shape),
        Convolution2D(16, (3, 3), activation='relu', padding='same'),
        Convolution2D(16, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Dropout(0.2),
        Convolution2D(32, (3, 3), activation='relu', padding='same'),
        Convolution2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Dropout(0.2),
        Convolution2D(64, (3, 3), activation='relu', padding='same'),
        Convolution2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Dropout(0.2),
        Convolution2D(128, (3, 3), activation='relu', padding='same'),
        Convolution2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Dropout(0.2),
        Convolution2D(256, (3, 3), activation='relu', padding='same'),
        Convolution2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(name='last'),
        # Flatten(),
        # Dense(1024, activation='relu'),

        # decoder
        # UpSampling2D(size=(2, 2)),
        Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2)),
        Dropout(0.2),
        Convolution2D(128, (3, 3), activation='relu', padding='same'),
        Convolution2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        # UpSampling2D(size=(2, 2)),
        Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2)),
        Dropout(0.2),
        Convolution2D(64, (3, 3), activation='relu', padding='same'),
        Convolution2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        # UpSampling2D(size=(2, 2)),
        Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2)),
        Dropout(0.2),
        Convolution2D(32, (3, 3), activation='relu', padding='same'),
        Convolution2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        # UpSampling2D(size=(2, 2)),

        # Convolution2D(8, (17, 17), strides=(20, 20)),
        Convolution2D(16, kernel_size=(3, 3), strides=(2, 2), activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Convolution2D(12, kernel_size=(3, 3), strides=(2, 2), activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Convolution2D(8, kernel_size=(1, 1), strides=(2, 2), activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Convolution2D(5, (1, 1), activation='softmax')
    ])


def auto_encoder_avg_pooling(shape):
    input_img = Input(shape=shape)  # adapt this if using `channels_first` image data format
    x = Lambda(lambda x: x / 255.0)(input_img)
    x = Convolution2D(30, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(30, (3, 3), activation='relu', padding='same')(x)
    x = AvgPool2D((2, 2), padding='same')(x)
    x = Convolution2D(30, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(30, (3, 3), activation='relu', padding='same')(x)
    x = AvgPool2D((2, 2), padding='same')(x)
    x = Convolution2D(30, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(30, (3, 3), activation='relu', padding='same')(x)
    x = AvgPool2D((2, 2), padding='same')(x)
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    x = GaussianNoise(0.05)(x)
    x = Convolution2D(30, (3, 3), activation='relu', padding='same')(x)

    x = Convolution2D(30, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(30, (3, 3), activation='relu', padding='same', name='encoder_output')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(30, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(30, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(30, (3, 3), activation='relu', padding="same")(x)
    x = Convolution2D(30, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(3, (3, 3), activation="sigmoid", padding='same')(x)
    decoded = Lambda(lambda x: x * 255.0)(decoded)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer="adadelta", loss=keras.losses.MSE, metrics=["accuracy"])
    return autoencoder



def auto_encoder_no_pooling(shape):
    input_img = Input(shape=shape)  # adapt this if using `channels_first` image data format
    x = Lambda(lambda x: x / 127.0)(input_img)
    x = Convolution2D(26, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(26, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = Convolution2D(26, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Convolution2D(26, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(26, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = Convolution2D(26, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Convolution2D(26, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(26, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    encoded = Convolution2D(26, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    x = Convolution2D(26, (3, 3), activation='relu', padding='same')(encoded)
    x = Convolution2D(26, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(26, (3, 3), name='encoder_output', activation='relu', padding='same')(x)

    x = Dropout(0.1)(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(26, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(26, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(26, (3, 3), activation='relu', padding="same")(x)
    x = Convolution2D(26, (3, 3), activation='relu')(x)
    x = Dropout(0.1)(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(3, (3, 3), activation="sigmoid", padding='same')(x)
    decoded = Lambda(lambda x: x * 127.0)(decoded)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer="adadelta", loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])
    return autoencoder


def auto_encoder(shape):
    input_img = Input(shape=shape)  # adapt this if using `channels_first` image data format
    x = Lambda(lambda x: x / 255.0)(input_img)
    x = Convolution2D(25, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(25, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Convolution2D(25, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(25, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Convolution2D(25, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(25, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    encoded = MaxPooling2D(strides=(2, 2), padding='same')(x)
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    x = Convolution2D(25, (3, 3), activation='relu', padding='same')(encoded)
    x = Convolution2D(25, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(25, (3, 3), name='encoder_output', activation='relu', padding='same')(x)

    x = Dropout(0.1)(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(25, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(25, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(25, (3, 3), activation='relu', padding="same")(x)
    x = Convolution2D(25, (3, 3), activation='relu')(x)
    x = Dropout(0.1)(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(3, (3, 3), activation="sigmoid", padding='same')(x)
    decoded = Lambda(lambda x: x * 255.0)(decoded)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer="adadelta", loss=keras.losses.MSE, metrics=["accuracy"])
    return autoencoder


def ourSemanticSegmentation(auto_encoder_path):
    auto_encoder = load_model(auto_encoder_path)
    for layer in auto_encoder.layers:
        layer.trainable = False

    x = auto_encoder.get_layer('encoder_output').output

    x = Convolution2D(30, (3, 3), activation="relu", padding="same", name='a1')(x)
    x = AvgPool2D((2, 2), padding="same", name='a2')(x)
    x = Convolution2D(32, (3, 3), activation="relu", padding="same", name='a3')(x)
    x = AvgPool2D((2, 2), padding="same", name='a4')(x)
    x = Convolution2D(32, (3, 3), activation="relu", padding="same", name='a5')(x)

    x = Convolution2D(32, (3, 3), activation="relu", padding="same", name='a6')(x)
    x = UpSampling2D((2, 2), name='a7')(x)
    x = Convolution2D(5, (3, 3), activation="relu", padding="same", name='a8')(x)

    model = Model(inputs=auto_encoder.input, outputs=x)
    model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])
    return model



def iou_better(actual, predicted):
    actual = K.abs(K.flatten(actual))
    predicted = K.abs(K.flatten(predicted))
    intersection = K.sum(actual * predicted)
    union = K.sum(actual) + K.sum(predicted) - intersection
    return intersection / union




if __name__ == "__main__":
    # model = auto_encoder_avg_pooling((372, 372, 3))
    model = ourSemanticSegmentation('../models/encoder_decoder/autoencoder_2018_09_24_17_50.model')
    model.summary()
    '''
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
    '''
