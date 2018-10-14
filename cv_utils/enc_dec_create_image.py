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


def create_image(model_path, images, batch_size=1):
    output_size = 24

    crops = [create_crops(image, 372, 372, 372 // 2, 372 // 2) for image in images]
    crops = np.concatenate(crops).reshape((-1, 372, 372, 3))
    height, width = calc_dim(images[0], 372, 372, 372 // 2, 372 // 2)
    dims = [calc_dim(image, 372, 372, 372 // 2, 372 // 2) for image in images]
    target = []
    # target = np.zeros((len(images), height * output_size, width * output_size, 1), dtype=np.uint8)
    model = load_model(model_path)

    test_generator = ImageDataGenerator().flow(x=np.array(crops), batch_size=batch_size, shuffle=False)

    predicts = model.predict_generator(test_generator,
                                       steps=len(crops) // batch_size,
                                       verbose=1,
                                       workers=8,
                                       use_multiprocessing=False)

    predicts = np.array(predicts).argmax(axis=-1).reshape(-1, output_size, output_size)

    # predicts = predicts.reshape((len(images) * height * width, output_size, output_size))

    i = 0
    for image_index in range(len(images)):
        height = dims[image_index][0]
        width = dims[image_index][1]
        tmp_img = np.zeros((height * output_size, width * output_size))
        for y in range(height):
            for x in range(width):
                tmp_img[y * output_size:(y + 1) * output_size, x * output_size:(x + 1) * output_size] = predicts[i]
                i += 1

        target.append(tmp_img)

    return target


def check(image_path):
    res = cv2.imread(image_path, 1)
    if res is None:
        print(image_path)

    return res


def create(src, model_path, save_path, mapping):
    paths = [image_path for image_path in glob.iglob(path.join(src, "*.JPG"))]
    print('There are', len(paths), 'paths')
    n = 40
    path_divides = [paths[i:min(i + n, len(paths))] for i in range(0, len(paths), n)]
    for path_list in path_divides:
        images = [check(image_path) for image_path in tqdm(path_list)]
        res = create_image(model_path, images, mapping)

        for result_image, i in zip(res, range(len(res))):
            tmp_path = path_list[i].replace("/", "\\")
            name = "\\".join(tmp_path.split('\\')[-2:])
            full_name = path.join(results_dir, name).replace("JPG", "txt")
            with open(full_name, 'w') as f:
                # print(full_name)
                json.dump(result_image.tolist(), f)
            # cv2.imwrite(full_name, result_image)


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

        '''
        for subdir2 in os.listdir(current_path):
            current_path2 = path.join(current_path, subdir2)
            save_path2 = path.join(save_path, subdir2)

            if not path.exists(save_path2):
                os.mkdir(save_path2)
        '''
