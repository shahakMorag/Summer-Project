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


def create_image(model_path, images, batch_size=1):
    output_size = 24

    crops = [create_crops(image, 372, 372, 372 // 2, 372 // 2) for image in images]
    # cv2.imshow("", crops[0])
    # cv2.waitKey(10000)
    crops = np.array(crops).reshape((-1, 372, 372, 3))
    height, width = calc_dim(images[0], 372, 372, 372 // 2, 372 // 2)
    target = np.zeros((len(images), height * output_size, width * output_size, 3), dtype=np.uint8)
    model = load_model(model_path)

    test_generator = ImageDataGenerator().flow(x=np.array(crops), batch_size=batch_size, shuffle=False)

    predicts = model.predict_generator(test_generator,
                                       steps=len(crops) // batch_size,
                                       verbose=1,
                                       workers=8,
                                       use_multiprocessing=False)

    predicts = np.array(predicts).reshape((len(images), -1, output_size * output_size, 5))
    predicts = predicts.argmax(axis=-1).reshape((len(images), -1))
    predicts = np.array([keys2img(predict, output_size, output_size, height * width) for predict in predicts])
    predicts = predicts.reshape((len(images), height, width, output_size, output_size, 3))

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


def create(src):
    paths = [image_path for image_path in glob.iglob(path.join(src, "*.JPG"))]
    print('There are', len(paths), 'paths')
    n = 20
    path_divides = [paths[i:min(i + n, len(paths))] for i in range(0, len(paths), n)]
    for path_list in path_divides:
        images = [check(image_path) for image_path in tqdm(path_list)]
        res = create_image("../models/encoder_decoder/semantic_seg_2018_09_25_10_4.model", images)
        for result_image, i in zip(res, range(len(res))):
            tmp_path = path_list[i].replace("/", "\\")
            name = "\\".join(tmp_path.split('\\')[-3:])
            full_name = path.join(results_dir, name)
            cv2.imwrite(full_name, result_image)


if __name__ == '__main__':
    results_dir = 'D:/results_encoder_decoder_new'
    main_dir = 'C:\Tomato_Classification_Project\Tomato_Classification_Project\Data\Demonstration_greenhouse_tomato_Hazera\Demonstration_greenhouse_tomato_Hazera'

    if not path.exists(results_dir):
        os.mkdir(results_dir)

    for subdir in os.listdir(main_dir):
        current_path = path.join(main_dir, subdir)
        save_path = path.join(results_dir, subdir)

        if not path.exists(save_path):
            os.mkdir(save_path)

        for subdir2 in os.listdir(current_path):
            current_path2 = path.join(current_path, subdir2)
            save_path2 = path.join(save_path, subdir2)

            if not path.exists(save_path2):
                os.mkdir(save_path2)

            create(current_path2)
