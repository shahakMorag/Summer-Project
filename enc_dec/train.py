import json

import cv2
import keras
import numpy as np
import datetime
import time

from nets import *
from keras.optimizers import Adam
from keras import backend as K
from random import sample
from encoder_decoder import *
from os import path


def get_start_date():
    return str(datetime.date.today()).replace('-', '_') + "_" + str(datetime.datetime.now().hour) + "_" + str(
        datetime.datetime.now().minute)


def load_data2(indexes, start, step, load_labels=False):
    features = np.zeros((step, 372, 372, 3), dtype=np.float32)
    labels = np.zeros((step, 24, 24, 5), dtype=np.float32)
    samples = indexes[start: start + step]

    for i, j in zip(samples, range(step)):
        feature = cv2.imread(
            'C:\Tomato_Classification_Project\Tomato_Classification_Project\cropped_data\sized_crop/' + str(
                i) + '.png')
        features[j] = feature
        if load_labels:
            with open('C:\Tomato_Classification_Project\Tomato_Classification_Project\encoder_decoder_train_set/' + str(
                    i) + '.txt') as f:
                label = json.loads(f.read())
            # label = np.reshape(label, (24, 24, 5))
            labels[j] = label
    return features, labels if load_labels else features


def load_data(start, stop, load_labels=False):
    features = np.zeros((stop - start, 372, 372, 3), dtype=np.float32)
    labels = np.zeros((stop - start, 24, 24, 5), dtype=np.float32)
    for i, j in zip(range(start, stop), range(stop - start)):
        feature = cv2.imread(
            'C:\Tomato_Classification_Project\Tomato_Classification_Project\cropped_data\sized_crop/' + str(
                i) + '.png')
        features[j] = feature
        if load_labels:
            with open('C:\Tomato_Classification_Project\Tomato_Classification_Project\encoder_decoder_train_set/' + str(
                    i) + '.txt') as f:
                label = json.loads(f.read())
            # label = np.reshape(label, (24, 24, 5))
            labels[j] = label
    return features, labels if load_labels else features


class Test(object):
    i = 0


def pipeline_generator(input_features, input_labels, batch_size, image_size, label_shape):
    while True:
        features = np.zeros((batch_size, image_size[1], image_size[0], 3), dtype=np.float32)
        labels = np.zeros((batch_size, label_shape[0], label_shape[1], label_shape[2]), dtype=np.float32)
        # samples = range() # sample(range(len(input_features)), batch_size)
        features = input_features[Test.i:Test.i + batch_size]
        labels = input_labels[Test.i:Test.i + batch_size]

        Test.i += batch_size
        Test.i %= len(input_features)
        '''
        for i,j in zip(samples, range(batch_size)):
            features[j] = input_features[i]
            labels[j] = input_labels[i]
        '''
        yield features, labels


def iou_better(actual, predicted):
    actual = K.abs(K.flatten(actual))
    predicted = K.abs(K.flatten(predicted))
    intersection = K.sum(actual * predicted)
    union = K.sum(actual) + K.sum(predicted) - intersection
    return intersection / union


def iou_simple(actual, predicted):
    actual = K.flatten(actual)
    predicted = K.flatten(predicted)
    return K.sum(actual * predicted) / (1.0 + K.sum(actual) + K.sum(predicted))


def val_loss(actual, predicted):
    return -iou_simple(actual, predicted)



# Training autoencoder
model = auto_encoder_avg_pooling((372,372,3))
model_path_auto = path.join("../models/encoder_decoder", "autoencoder_" + get_start_date() + ".model")
print("saving model to:", model_path_auto)
start = 0
step = 1500  # 1500
n_max = 280380
ordered = np.random.permutation(n_max)
for i in range(186): # 125
    print("number:", str(i))
    features, labels = load_data2(ordered, start, step)
    # features, labels = load_data(157, 158)
    generator = pipeline_generator(features, labels, 10, (372, 372), (372, 372, 3))
    model.fit_generator(generator, epochs=10, verbose=1, steps_per_epoch=150, workers=8)
    start += step


model.save(model_path_auto)
model = None
time.sleep(10)

# Training encoder decoder
model = ourSemanticSegmentation(model_path_auto)
model_path = path.join("../models/encoder_decoder", "semantic_seg_" + get_start_date() + ".model")
print("saving model to:", model_path)
start = 0
step = 2500  # 2500
n_max = 105320
ordered = np.random.permutation(n_max)
for i in range(200):  # 200
    print("number:", str(i))
    features, labels = load_data2(ordered, start, step, load_labels=True)
    generator = pipeline_generator(features, labels, 25, (372, 372), (24, 24, 5))
    model.fit_generator(generator, epochs=15, verbose=1, steps_per_epoch=100, workers=8)
    start += step
    start %= n_max

model.save(model_path)
