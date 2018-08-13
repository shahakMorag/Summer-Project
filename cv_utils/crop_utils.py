import cv2
import numpy as np

from NN.model import custom_network

im = img = cv2.imread('../test/image transformations/eyal.png', 1)
# im = cv2.resize(im,)
# im = im.resize((500, 500))

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


# 0 - bad leaf
# 1 - fruit
# 2 - leaf
# 3 - other
# 4 - stem
keys = [0, 1, 2, 3, 4]
colors = [[0, 0, 255], [255, 0, 0], [0, 255, 0], [0, 255, 255], [255, 255, 0]]

dict = dict(zip(keys, colors))


def keys2img(vals, height, width):
    res = []
    for item in vals:
        res.append(dict.get(item))

    return np.reshape(res, (height, width, 3))

list = createCrops(im, step, step, crop_x, crop_y)
m = apply_classification(list)
imcv = keys2img(m, 11, 3)

cv2.imshow("shem sel hahalon", imcv)
cv2.waitKey(0)

#crops_show(list)
