import glob
from os import path

import cv2
import keras
import numpy as np
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from crop_utils import create_crops, calc_dim, keys2img
from tqdm import tqdm
import os
import json
import re


def create_image(model_path, images, mapping, batch_size=1):
    output_size = 24

    crops = [create_crops(image, 372, 372, 372 // 2, 372 // 2) for image in images]
    crops = np.array(crops).reshape((-1, 372, 372, 3))
    height, width = calc_dim(images[0], 372, 372, 372 // 2, 372 // 2)
    target = np.zeros((len(images), height * output_size, width * output_size, 1), dtype=np.uint8)
    model = load_model(model_path)

    test_generator = ImageDataGenerator().flow(x=np.array(crops), batch_size=batch_size, shuffle=False)

    predicts = model.predict_generator(test_generator,
                                       steps=len(crops) // batch_size,
                                       verbose=1,
                                       workers=8,
                                       use_multiprocessing=False)

    predicts = np.array(predicts).reshape((len(images), -1, output_size * output_size, 5))
    predicts = predicts.argmax(axis=-1).reshape((len(images), -1))
    predicts = predicts.reshape((len(images), height, width, output_size, output_size, 1))

    for image_index in range(len(images)):
        for y in range(height):
            for x in range(width):
                target[image_index, y * output_size:(y + 1) * output_size, x * output_size:(x + 1) * output_size] = \
                predicts[image_index, y, x]

    return target


def check(image_path):
    res = cv2.imread(image_path, 1)
    if res is None:
        print(image_path)

    return res


def create(src, model_path, save_path, mapping):
    paths = [image_path for image_path in glob.iglob(path.join(src, "*.JPG"))]
    print('There are', len(paths), 'paths')
    n = 1
    path_divides = [paths[i:min(i + n, len(paths))] for i in range(0, len(paths), n)]
    for path_list in path_divides:
        images = [check(image_path) for image_path in tqdm(path_list)]
        res = create_image(model_path, images, mapping)
        for result_image, i in zip(res, range(len(res))):
            orig_dir, file_name = path.split(paths)
            file_name, extension = path.splitext(file_name)

            full_name = path.join(*[save_path, file_name + 'txt'])
            with open(full_name, 'w') as f:
                json.dump(result_image.tolist(), f)


def parse_color_mapping(mapping_description):
    res = {}
    for k, v in mapping_description.items():
        b, g, r = re.search(r'(\d+),(\d+),(\d+)', v).group(1, 2, 3)
        res[int(k)] = np.array([b, g, r], dtype=np.uint8)

    return res


if __name__ == '__main__':
    with open('encDecCreateImageConfig.json') as fp:
        args = json.load(fp)

    results_dir = args["destPath"]
    main_dir = args["sourcePath"]

    mapping = parse_color_mapping(args["mappingColors"])

    if not path.exists(results_dir):
        os.mkdir(results_dir)

    for subdir in os.listdir(main_dir):
        current_path = path.join(main_dir, subdir)
        save_path = path.join(results_dir, subdir)

        if not path.exists(save_path):
            os.mkdir(save_path)

        create(current_path, args["modelPath"], save_path, mapping)
