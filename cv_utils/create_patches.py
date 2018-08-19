import os
import cv2
import glob
import numpy as np
import cv_utils.transformations
from cv_utils.transformations import create_perspective_patches, create_rotated_patches, correct_gamma

path = "D:\Tomato_Classification_Project\Patches\Patches"


def get_pictures(dir):
    for filename in glob.glob(dir + '/*.png'):
        im = cv2.imread(filename, 1)
        pers = create_perspective_patches(im)
        rot = create_rotated_patches(im)

        per_g, rot_g = [], []

        for img in pers:
            tmp = correct_gamma(img)
            per_g.append(tmp)

        for img in rot:
            tmp = correct_gamma(img)
            rot_g.append(tmp)

        i = 0
        for img in per_g:
            filename = filename.replace('.png', '').replace('\\', '/')

            path = filename + '_per_' + repr(i) + '.png'
            print('saving to ' + path)
            print(cv2.imwrite(path, img))
            i += 1

        i = 0
        for img in rot_g:
            filename = filename.replace('.png', '').replace('\\', '/')

            path = filename + '_rot_' + repr(i) + '.png'
            print('saving to ' + path)
            print(cv2.imwrite(path, img))
            i += 1


# get_pictures('D:/New folder')


get_pictures('D:\Tomato_Classification_Project_5_iter\Patches\Patches\patches_size_128_skip_32_categories_5/bad_leaf')
get_pictures('D:\Tomato_Classification_Project_5_iter\Patches\Patches\patches_size_128_skip_32_categories_5/fruit')
get_pictures('D:\Tomato_Classification_Project_5_iter\Patches\Patches\patches_size_128_skip_32_categories_5/leaf')
get_pictures('D:\Tomato_Classification_Project_5_iter\Patches\Patches\patches_size_128_skip_32_categories_5/other')
get_pictures('D:\Tomato_Classification_Project_5_iter\Patches\Patches\patches_size_128_skip_32_categories_5/stem')
