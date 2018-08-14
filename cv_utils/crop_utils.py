import cv2
import numpy as np

from NN.model import custom_network

im = img = cv2.imread('../test/image transformations/eyal.png', 1)

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


def crops_show(im_list):
    for crop in im_list:
        cv2.imshow("ddsa", crop)
        cv2.waitKey(1000)


def apply_classification(image_list):
    model = custom_network()
    model.load('../NN/first.model')
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

    return np.reshape(res, (height, width, 3))


# list = createCrops(im, step, step, crop_x, crop_y)
# m = apply_classification(list)

re = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
m = re * 25000

imcv = keys2img(m, 500, 1000)

cv2.imshow("shem sel hahalon", imcv)
cv2.waitKey(0)

#crops_show(list)
