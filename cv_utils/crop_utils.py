import tensorflow as tf
import cv2
import numpy as np

from NN.AntiRectifier import Antirectifier
from NN.model import custom_network, changed_model

im = img = cv2.imread('../test/image transformations/IMG_5562.JPG', 1)

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
    # model = changed_model()
    model = tf.keras.models.load_model('../NN/first.model', custom_objects={'Antirectifier': Antirectifier})
    lst = []
    for im in image_list:
        im = im.reshape(-1, 128, 128, 3)
        res = model.predict(im)
        lst.append(np.argmax(res))
    return lst


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
    res = []
    for item in vals:
        res.append(dict.get(item))

    return np.reshape(res, (int(height), int(width), 3))


# We better trust practical calculations...
new_height, new_width = calc_dim(im, step, step, crop_x, crop_y)

list = createCrops(im, step, step, crop_x, crop_y)
m = apply_classification(list)

imcv = keys2img(m, new_height, new_width)

cv2.imshow("shem sel hahalon", imcv)
cv2.waitKey(0)

#crops_show(list)
