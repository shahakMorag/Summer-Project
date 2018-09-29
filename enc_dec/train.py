import datetime
import json
import os
import time
from os import path

import cv2
import numpy as np
from encoder_decoder import *
import argparse

def get_start_date():
    return str(datetime.date.today()).replace('-', '_') + "_" + str(datetime.datetime.now().hour) + "_" + str(
        datetime.datetime.now().minute)


def load_data2(indexes, start, step, crops_dir, ground_truth_dir=None, load_labels=False):
    features = np.zeros((step, 372, 372, 3), dtype=np.float32)
    labels = np.zeros((step, 24, 24, 5), dtype=np.float32)
    samples = indexes[start: start + step]

    for i, j in zip(samples, range(step)):
        features[j] = cv2.imread(path.join(crops_dir, str(i) + ".png"))
        if load_labels:
            with open(path.join(ground_truth_dir, str(i) + ".txt")) as f:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-classifier", required=True)
    parser.add_argument("-ground_truth_dir", required=True)
    parser.add_argument("-crops_dir", required=True)
    parser.add_argument("-auto_encoder_n_max", required=True, type=int)
    parser.add_argument("-encoder_decoder_n_max", required=True, type=int)
    parser.add_argument("-auto_encoder_training_epochs", type=int)
    parser.add_argument("-encoder_decoder_training_epochs", type=int)
    args = parser.parse_args()
    dir_to_save = path.join(*["models", "encoder_decoder", args.classifier])
    if not path.exists(dir_to_save):
        os.mkdir(dir_to_save)

    # Training autoencoder
    model = auto_encoder_avg_pooling((372, 372, 3))
    model_path_auto = path.join(*[dir_to_save, "autoencoder_" + get_start_date() + ".model"])
    print("saving model to:", model_path_auto)
    start = 0
    step = 1500
    batch_size = 10
    ordered = np.random.permutation(args.auto_encoder_n_max)
    total_epochs = args.auto_encoder_training_epochs if args.auto_encoder_training_epochs is not None else args.auto_encoder_n_max // step
    for i in range(total_epochs):

        features, labels = load_data2(ordered, start, step, args.crops_dir)
        # features, labels = load_data(157, 158)
        generator = pipeline_generator(features, labels, batch_size, (372, 372), (372, 372, 3))
        model.fit_generator(generator, epochs=10, verbose=1, steps_per_epoch=step // batch_size, workers=8)
        start += step

    model.save(model_path_auto)
    model = None
    time.sleep(10)

    # Training encoder decoder
    model = ourSemanticSegmentation(model_path_auto)
    model_path = path.join(*[dir_to_save, "sematnic_segmentation_" + get_start_date() + ".model"])
    print("saving model to:", model_path)
    start = 0
    step = 2500
    ordered = np.random.permutation(args.encoder_decoder_n_max)
    total_epochs = args.encoder_decoder_training_epochs if args.encoder_decoder_training_epochs is not None else 2 * args.encoder_decoder_n_max // step
    for i in range(total_epochs):
        print("number:", str(i))
        features, labels = load_data2(ordered, start, step, args.crops_dir, args.ground_truth_dir, load_labels=True)
        generator = pipeline_generator(features, labels, 25, (372, 372), (24, 24, 5))
        model.fit_generator(generator, epochs=15, verbose=1, steps_per_epoch=100, workers=8)
        start += step
        start %= args.encoder_decoder_n_max

    model.save(model_path)
