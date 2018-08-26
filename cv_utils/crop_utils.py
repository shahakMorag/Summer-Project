import cv2
import numpy as np
import time
from transformations import get_relative_brightness, correct_gamma
import keras
from keras_preprocessing.image import ImageDataGenerator

from keras.models import load_model
'''
import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
'''
im = cv2.imread('../test/image transformations/IMG_5562.JPG', 1)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

step = 13
radius_x = 64
radius_y = 64

non_green_radius_x = 15
non_green_radius_y = 15

scale_after = 2
total_start_time = time.time()

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
            '''
            if not is_green(cropped):
                cropped = i_img[y_mid - non_green_radius_y:y_mid + non_green_radius_y,
                          x_mid - non_green_radius_x:x_mid + non_green_radius_x]
            else:
                cropped = i_img[y_mid - radius_y:y_mid + radius_y, x_mid - radius_x:x_mid + radius_x]

            cropped = cv2.resize(cropped, (128, 128))
            '''
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


def apply_classification(image_list, batch_size=1, model_path='../models/mobilenet/2018_08_25_17_6_500_epochs.model'):
    start_time = time.time()
    print("Applying classification...")
    model = load_model(model_path)

    test_generator = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input) \
        .flow(x=np.array(image_list),
              batch_size=batch_size,
              shuffle=False,
              )

    predicts = model.predict_generator(test_generator,
                                       steps=len(image_list),
                                       verbose=1,
                                       workers=16)

    tags = predicts.argmax(axis=1)
    end_time = time.time()
    d_time = end_time - start_time
    print("Classification took " + repr(d_time) + " seconds")
    return np.array(tags).flatten()


maps = dict(zip([0,1,2],[0,2,3]))
def fix_classes(m, m2, leafs_indexes):
    i = 0
    while i < len(m2):
        # the 2 is because the numbers are only in [0,1]
        m[leafs_indexes[i]] = maps.__getitem__(m2[i])
        i += 1



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
m = apply_classification(crops_list)
leafs_indexes = np.where(np.isin(m, [0, 2, 3]))[0]
leafs_crop = crops_list[leafs_indexes.tolist()]
m2 = apply_classification(leafs_crop, model_path="../models/mobilenet/2018_08_26_18_20_20_epochs_leaf.model")
fix_classes(m, m2, leafs_indexes)

imcv = keys2img(m, new_height, new_width)
imcv = cv2.resize(imcv, None, fx=scale_after, fy=scale_after)
cv2.imshow("shem sel hahalon", imcv)
cv2.waitKey(0)
