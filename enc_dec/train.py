import json

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import cv2
from mobilenet2 import encoder_decoder
import keras

def load_data(images):
    features = np.zeros((images, 500, 500, 3), dtype=np.uint8)
    labels = np.zeros((images, 24, 24, 5), dtype=np.float32)
    for i in range(images):
        feature = cv2.imread(
            'C:\Tomato_Classification_Project\Tomato_Classification_Project\cropped_data\size_500_stride_500/' + str(i) + '.png')
        features[i] = feature
        with open('C:\Tomato_Classification_Project\Tomato_Classification_Project\encoder_decoder_train_set/' + str(i) + '.txt') as f:
            label = json.loads(f.read())
        label = np.reshape(label, (24, 24, 5))
        labels[i] = label
    return features, labels


def pipeline_generator(input_features, input_labels, batch_size, image_size):
    data_length = len(input_features)
    while True:
        i = 0
        features = np.zeros((batch_size, image_size[1], image_size[0], 3), dtype=np.uint8)
        labels = np.zeros((batch_size, 24, 24, 5), dtype=np.float32)
        while i < batch_size:
            index = np.random.randint(data_length)
            feature = input_features[index]
            label = input_labels[index]
            features[i] = feature
            labels[i] = label
            i += 1
        yield features, labels


# def iou_better(actual, predicted):
#     actual = K.abs(K.flatten(actual))
#     predicted = K.abs(K.flatten(predicted))
#     intersection = K.sum(actual * predicted)
#     union = K.sum(actual) + K.sum(predicted) - intersection
#     return intersection / union
#
#
# def iou_simple(actual, predicted):
#     actual = K.flatten(actual)
#     predicted = K.flatten(predicted)
#     return K.sum(actual * predicted) / (1.0 + K.sum(actual) + K.sum(predicted))
#
#
# def val_loss(actual, predicted):
#     return -iou_simple(actual, predicted)


features, labels = load_data(10000)
model = encoder_decoder((500, 500, 3))
generator = pipeline_generator(features, labels, 1, (500, 500))
optimizer = Adam(lr=1.0e-5)

model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])
model.fit_generator(generator, samples_per_epoch=10000, nb_epoch=10,  verbose=1)
model.save("../models/encoder_decoder/m.model")