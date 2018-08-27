import cv2
import numpy as np
import time
from transformations import get_relative_brightness, correct_gamma
import keras
from keras_preprocessing.image import ImageDataGenerator

from keras.models import load_model
'''
import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
'''
im = cv2.imread('../test/image transformations/IMG_5562.JPG', 1)
im = cv2.resize(im, None, fx=(1/1.3), fy=(1/1.3))
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

step = 20
radius_x = 64
radius_y = 64

non_green_radius_x = 15
non_green_radius_y = 15

scale_after = 3
total_start_time = time.time()

_res = [None, None]
m = None


def is_green(i_img):
    b, g, r = cv2.split(i_img)
    b_b = get_relative_brightness(b, 1)
    b_g = get_relative_brightness(g, 1)
    b_r = get_relative_brightness(r, 1)
    return (b_b + 5 / 256) < b_g and (b_r + 5 / 256) < b_g


def is_red(i_img):
    b, g, r = cv2.split(i_img)
    b_b = get_relative_brightness(b, 1)
    b_g = get_relative_brightness(g, 1)
    b_r = get_relative_brightness(r, 1)
    return (b_g + 5 / 256) < b_r and (b_b + 5 / 256) < b_r


def create_crops(i_img, step_x, step_y, radius_x, radius_y):
    res = []
    height, width = i_img.shape[:2]

    for y_mid in range(radius_y, height - radius_y, step_y):
        for x_mid in range(radius_x, width - radius_x, step_x):
            cropped = i_img[y_mid - radius_y:y_mid + radius_y, x_mid - radius_x:x_mid + radius_x]
            res.append(cropped)

    return res


def calc_dim(im, step_x, step_y, radius_x, radius_y):
    im_w, im_h = np.size(im, 1), np.size(im, 0)
    height, width = im.shape[:2]

    h = 0
    w = 0

    for up in range(radius_y, height - radius_y, step_y):
        h += 1

    for left in range(radius_x, width - radius_x, step_x):
        w += 1

    return h, w


def crops_show(im_list):
    for crop in im_list:
        cv2.imshow("ddsa", crop)
        cv2.waitKey(1000)


def apply_classification(image_list, model_path='../models/mobilenet/2018_08_27_16_53_5_epochs_class_all.model'):
    start_time = time.time()
    print("Applying classification...")

    model = load_model(model_path)

    test_generator = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input) \
        .flow(x=np.array(image_list),
              batch_size=1,
              shuffle=False)

    predicts = model.predict_generator(test_generator,
                                       steps=len(image_list),
                                       verbose=1,
                                       workers=1,
                                       use_multiprocessing=False)

    tags = predicts.argmax(axis=1)
    end_time = time.time()
    d_time = end_time - start_time
    print("Classification took " + repr(d_time) + " seconds")
    return np.array(tags).flatten()


maps = dict(zip([0, 1, 2], [0, 2, 3]))


def fix_classes(m, m2, leafs_indexes):
    i = 0
    while i < len(m2):
        # the 2 is because the numbers are only in [0,1]
        m[leafs_indexes[i]] = maps.__getitem__(m2[i])
        i += 1


# We assume that the images in the same size
def blend_two_images(neural_net_image, original_image, alpha=1.0):
    original_image = original_image[radius_y:-radius_y, radius_x:-radius_x]
    original_image = cv2.resize(original_image, tuple(neural_net_image.shape[1::-1]))

    beta = (1 - alpha)
    dst = cv2.addWeighted(neural_net_image, alpha, original_image, beta, 0.0)
    cv2.imshow("halon", dst)
    cv2.waitKey(0)


# 0 - bad leaf - blue    - [255, 0, 0]
# 1 - fruit    - red     - [35,28,229]
# 2 - leaf     - green   - [0, 255, 0]
# 3 - other    - brown   - [0,255,239]
# 4 - stem - dark green  - [16,64,4]
keys = [0, 1, 2, 3, 4]
colors = np.array([[255, 0, 0],
                   [35, 28, 229],
                   [0, 255, 0],
                   [0, 255, 239],
                   [16, 64, 4]]).astype(np.uint8)
dict = dict(zip(keys, colors))


# [0, 0, 255] - red
# [0, 255, 0] - green
# [255, 0, 0] - blue

def keys2img(vals, height, width):
    print("Creating image...")
    start_time = time.time()
    res = []
    for item in vals:
        res.append(dict.get(item))

    end_time = time.time()
    d_time = end_time - start_time
    print("Image creation took " + repr(d_time) + " seconds")

    return np.reshape(res, (int(height), int(width), 3))


# We better trust practical calculations...
new_height, new_width = calc_dim(im, step, step, radius_x, radius_y)

crops_list = np.array(create_crops(im, step, step, radius_x, radius_y))
# crops_list_first_half = crops_list[:int(len(crops_list) / 4)]
# crops_list_second_half = crops_list[int(len(crops_list) / 2):]
m = apply_classification(crops_list)

imcv = keys2img(m, new_height, new_width)
imcv = cv2.resize(imcv, None, fx=scale_after, fy=scale_after)
blend_two_images(imcv, im, alpha=0.7)
