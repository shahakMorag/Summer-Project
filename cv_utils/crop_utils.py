import tensorflow as tf
import cv2
import numpy as np
import time

from NN.AntiRectifier import Antirectifier
from keras.models import load_model

im = img = cv2.imread('../test/image transformations/IMG_0781.JPG', 1)
im = im.astype('float32')
im /= 256

step = 16
crop_x = 128
crop_y = 128


def createCrops(im, step_x, step_y, crop_x, crop_y):
    res = []
    height, width = im.shape[:2]
    for up in range(0, height - crop_y, step_y):
        for left in range(0, width - crop_x, step_x):
            cropped = im[up:up + crop_y, left:left + crop_x]
            res.append(cropped)

    return res


def calc_dim(im, step_x, step_y, crop_x, crop_y):
    im_w, im_h = np.size(im, 1), np.size(im, 0)
    height, width = im.shape[:2]

    h = 0
    w = 0

    for up in range(0, height - crop_y, step_y):
        h += 1

    for left in range(0, width - crop_x, step_x):
        w += 1

    return h, w

def crops_show(im_list):
    for crop in im_list:
        cv2.imshow("ddsa", crop)
        cv2.waitKey(1000)


def apply_classification(image_list):
    print("Applying classification...")
    start_time = time.time()
    # model = changed_model()
    model = load_model('../NN/first.model')
    lst = model.predict(np.array(image_list), verbose=1)
    tags = lst.argmax(axis=1)

    end_time = time.time()
    d_time = end_time - start_time
    print("Classification took " + repr(d_time) + "seconds")
    return tags


# 0 - bad leaf - blue    - [255, 0, 0]
# 1 - fruit    - red     - [9,1,146]
# 2 - leaf     - green   - [0, 255, 0]
# 3 - other    - brown   - [0,25,74]
# 4 - stem - dark green  - [9,42,1]
keys = [0, 1, 2, 3, 4]
colors = np.array([[255, 0, 0], [9, 1, 146], [0, 255, 0], [0, 25, 74], [9, 42, 1]]).astype(np.uint8)
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
new_height, new_width = calc_dim(im, step, step, crop_x, crop_y)

list = createCrops(im, step, step, crop_x, crop_y)
m = apply_classification(list)

imcv = keys2img(m, new_height, new_width)

cv2.imshow("shem sel hahalon", imcv)
cv2.waitKey(0)

#crops_show(list)
