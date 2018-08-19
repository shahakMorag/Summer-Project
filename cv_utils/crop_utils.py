import cv2
import numpy as np
import time
from cv_utils.transformations import get_relative_brightness, correct_gamma

from keras.models import load_model

im = img = cv2.imread('../test/image transformations/IMG_5562.JPG', 1)
im = im.astype('float32')
im /= 256

step = 12
radius_x = 64
radius_y = 80

non_green_radius_x = 15
non_green_radius_y = 15


def is_green(im):
    b, g, r = cv2.split(im)
    b_b = get_relative_brightness(b, 1)
    b_g = get_relative_brightness(g, 1)
    b_r = get_relative_brightness(r, 1)

    is_green = (b_b + 5 / 256) < b_g and (b_r + 5 / 256) < b_g

    return is_green


def createCrops(im, step_x, step_y, radius_x, radius_y):
    res = []
    height, width = im.shape[:2]
    for y_mid in range(radius_y, height - radius_y, step_y):
        for x_mid in range(radius_x, width - radius_x, step_x):
            cropped = im[y_mid - radius_y:y_mid + radius_y, x_mid - radius_x:x_mid + radius_x]
            if not is_green(cropped):
                cropped = im[y_mid - non_green_radius_y:y_mid + non_green_radius_y,
                          x_mid - non_green_radius_x:x_mid + non_green_radius_x]
            cropped = cv2.resize(cropped, (128, 128))
            res.append(correct_gamma(cropped))

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


def apply_classification(image_list):
    print("Applying classification...")
    start_time = time.time()
    # model = changed_model()
    model = load_model('../NN/fifth.model')
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
new_height, new_width = calc_dim(im, step, step, radius_x, radius_y)

list = createCrops(im, step, step, radius_x, radius_y)
m = apply_classification(list)

imcv = keys2img(m, new_height, new_width)

cv2.imshow("shem sel hahalon", imcv)
cv2.waitKey(0)

#crops_show(list)

'''
im = img = cv2.imread('../test/image transformations/000101.png', 1)
is_green(im)
'''
