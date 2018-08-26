import os

import cv2
import glob
import numpy as np


def make_one_hot(lst, num_classes):
    tmp = np.array(lst, dtype=np.int32)
    tmp2 = np.zeros((tmp.shape[0], num_classes), dtype=np.float32)
    tmp2[np.arange(tmp.shape[0]), tmp] = 1
    return tmp2

'''
def get_pictures(dir):
    images = np.empty([limit,128,128,3], dtype=np.uint8)
    i = 0

    while limit > 0:
        for filename in glob.glob(dir + '\*.png'):
            if start > 0:
                start -= 1
                continue

            images[i] = cv2.imread(filename, 1)
            i += 1
            limit -= 1
            if limit == 0:
                break

    return images
'''


def get_pictures(dir):
    count = len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
    images = np.empty([count,128,128,3], dtype=np.uint8)
    i = 0

    for filename in os.listdir(dir):
        dest_name = dir + '//' + filename
        tmp = cv2.imread(dest_name, 1)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        images[i] = tmp
        i += 1

    return images, count


def make_inputs(path, num_classes=5):
    print("Making imputs:")
    X, count = get_pictures(path + "\\bad_leaf")
    Y = [0] * count

    tmp, count = get_pictures(path + "\\fruit")
    X = np.concatenate((X, tmp))
    Y += [1] * count

    tmp, count = get_pictures(path + "\\leaf")
    X = np.concatenate((X, tmp))
    Y += [2] * count

    tmp, count = get_pictures(path + "\\other")
    X = np.concatenate((X, tmp))
    Y += [3] * count

    tmp, count = get_pictures(path + "\\stem")
    X = np.concatenate((X, tmp))
    Y += [4] * count

    return X, Y
