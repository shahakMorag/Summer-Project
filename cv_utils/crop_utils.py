import cv2
import numpy as np
import time
from transformations import get_relative_brightness, correct_gamma
import keras
from keras_preprocessing.image import ImageDataGenerator

from keras.models import load_model

im = img = cv2.imread('../test/image transformations/IMG_5562.JPG', 1)

step = 20
radius_x = 64
radius_y = 64

non_green_radius_x = 15
non_green_radius_y = 15


def is_green(i_img):
    b, g, r = cv2.split(i_img)
    b_b = get_relative_brightness(b, 1)
    b_g = get_relative_brightness(g, 1)
    b_r = get_relative_brightness(r, 1)
    return (b_b + 5 / 256) < b_g and (b_r + 5 / 256) < b_g


def create_crops(i_img, step_x, step_y, radius_x, radius_y):
    res = []
    height, width = i_img.shape[:2]

    for y_mid in range(radius_y, height - radius_y, step_y):
        for x_mid in range(radius_x, width - radius_x, step_x):
            cropped = i_img[y_mid - radius_y:y_mid + radius_y, x_mid - radius_x:x_mid + radius_x]
            if not is_green(cropped):
                cropped = i_img[y_mid - non_green_radius_y:y_mid + non_green_radius_y,
                          x_mid - non_green_radius_x:x_mid + non_green_radius_x]
            else:
                cropped = i_img[y_mid - radius_y:y_mid + radius_y, x_mid - radius_x:x_mid + radius_x]

            cropped = cv2.resize(cropped, (128, 128))
            # corrected = correct_gamma(cropped)
            res.append(np.array(cropped, dtype=np.float32))

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
    '''
    image_list = np.array(image_list)
    image_list /= 255
    print("Applying classification...")
    start_time = time.time()
    # model = changed_model()
    model = load_model('../NN/test.model')
    lst = model.predict(np.array(image_list), verbose=1, rescale=)
    tags = lst.argmax(axis=1)

    end_time = time.time()
    d_time = end_time - start_time
    print("Classification took " + repr(d_time) + "seconds")
    return tags
    '''
    start_time = time.time()
    print("Applying classification...")
    model = load_model('../NN/20-8-18-1.model')

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow(
        x=np.array(image_list),
        batch_size=1,
        shuffle=False)

    nb_samples = len(image_list)
    predicts = model.predict_generator(test_generator,
                                       steps=nb_samples,
                                       verbose=1,
                                       workers=8)

    tags = predicts.argmax(axis=1)
    end_time = time.time()
    d_time = end_time - start_time
    print("Classification took " + repr(d_time) + "seconds")
    return tags

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

list = create_crops(im, step, step, radius_x, radius_y)
m = apply_classification(list)

imcv = keys2img(m, new_height, new_width)

cv2.imshow("shem sel hahalon", imcv)
cv2.waitKey(0)

#crops_show(list)

'''
im = img = cv2.imread('../test/image transformations/000101.png', 1)
is_green(im)
'''
