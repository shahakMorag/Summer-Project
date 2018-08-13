import cv2
import numpy as np

from NN.model import custom_network

im = img = cv2.imread('../test/image transformations/eyal.png', 1)
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
    model.load('first.model')
    return map(lambda x: np.argmax(x), map(lambda x: model.predict(x), image_list))



list = createCrops(im, step, step, crop_x, crop_y)
m = apply_classification(list)

crops_show(list)
