import os
import cv2
import glob
import numpy as np
import cv_utils.transformations
from cv_utils.transformations import create_perspective_patches, create_rotated_patches, correct_gamma

path = "D:\Tomato_Classification_Project\Patches\Patches"


def get_pictures(dir):
    images = []
    res = []

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
            cv2.imwrite(path, img)
            i += 1

    return res


get_pictures('D:/New folder')
