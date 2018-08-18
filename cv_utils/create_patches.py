import os
import cv2
import glob
import numpy as np
import cv_utils.transformations

path = "D:\Tomato_Classification_Project\Patches\Patches"


def get_pictures(dir):
    images = []
    res = []

    for filename in glob.glob(dir + '/*.png'):
        images.append(cv2.imread(filename, 1))

    return res


def save_batch(batch, path_to_dir):
    i = 0
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)

    for img in batch:
        cv2.imwrite(path_to_dir + repr(i) + '.png')


def make_patches():
    temp = get_pictures(path + "\patches_size_128_skip_32_categories_5/bad_leaf")


make_patches()
